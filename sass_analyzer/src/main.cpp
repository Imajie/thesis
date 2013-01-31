#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <cstdint>

#include "inst_types.h"

//#define DEBUG

using namespace std;

typedef struct MLP_ent{
	string reg;
	int load_idx;
	int use_idx;
	int load_count;

	MLP_ent( string reg = "", int load_idx = -1 ) 
		: reg(reg), load_idx(load_idx), use_idx(-1), load_count(0) {}
} MLP_entry;

// turn string into an instruction
void parse_instruction( instruction &inst, string s );

// parse out all dependencies of the source operand src
vector<string> extract_srcs( string src );

// output program flow as a digraph
void output_program_graph( string fileName, std::map<int, basic_block> &program );

int main( int argc, char **argv )
{
	ifstream asmFile;

	// open sass file
	if( argc == 2 )
	{
		asmFile.open(argv[1], ifstream::in);
	}
	else
	{
		cerr << "Usage: " << argv[0] << " file.asm.proc" << endl;
		exit(EXIT_FAILURE);
	}

	// build map of addr -> line
	map<int, instruction> addrToCode;
	vector<int> addrs;

	vector<int> startOfBlocks;

	while( asmFile.good() )
	{
		string line;
		instruction inst;

		getline(asmFile, line);

		if( line.length() > 0 )
		{
			// get assembly
			string code = line.substr(line.find("\t")+1);

			// get address
			stringstream num(line.substr(2, line.find("\t")-2-2));
			uint32_t addr;
			num >> hex >> addr;

			// create instruction
			inst.addr = addr;

			parse_instruction(inst, code);

			// add to map
			addrToCode[addr] = inst;
			addrs.push_back(addr);

			// check if start of new basic block
			if( (inst.cond.isCondition ^ addrToCode[addrs[addrs.size()-2]].cond.isCondition ) ||		// conditionality toggle
				(
				 (inst.cond.isCondition && addrToCode[addrs[addrs.size()-2]].cond.isCondition ) && 		// OR (both conditional AND
				 (
				  (inst.cond.reg[inst.cond.reg.size()-1] != addrToCode[addrs[addrs.size()-2]].cond.reg[addrToCode[addrs[addrs.size()-2]].cond.reg.size()-1] ) ||
				  (inst.cond.reg[0] == '!' ^ addrToCode[addrs[addrs.size()-2]].cond.reg[0] == '!')	// (different registers OR opposite test) )
				 ) 
				)
			  )
			{
				// start/end of conditional code execution
				startOfBlocks.push_back( addr );
			}

			if( inst.opcode == opcode_type::Branch )
			{
				// BRA instruction
				uint32_t jumpAddr;
				stringstream num( inst.dest.location.substr(2) ); // strip 0x

				num >> hex >> jumpAddr;
				startOfBlocks.push_back( jumpAddr );
			}
		}
	}

	// sort the block starting addresses
	startOfBlocks.push_back(0x0000);
	sort(startOfBlocks.begin(), startOfBlocks.end());
	auto it = unique(startOfBlocks.begin(), startOfBlocks.end());
	startOfBlocks.resize( it - startOfBlocks.begin() );

#ifdef DEBUG
	cout << "Blocks start:" << endl;
	for( auto a : startOfBlocks )
	{
		cout << hex << a << endl;
	}
#endif

	map<int, basic_block > blockCode;
	map<int, int> addrToBlock;
	// build up all instructions in each basic block, for ease of searching later
	int curBlock = 0;

	basic_block blockCodeTemp;
	for( int idx = 0; idx < addrs.size(); idx++ )
	{
		addrToBlock[addrs[idx]] = curBlock;
		blockCodeTemp.insts.push_back(addrToCode[addrs[idx]]);

		if( curBlock <= startOfBlocks.size()-1 && idx <= addrs.size()-2
				&& addrs[idx+1] >= startOfBlocks[curBlock+1] || idx == addrs.size() - 1)
		{
#ifdef DEBUG
			cout << "End of block - " << addrs[idx] << endl;
#endif
			blockCode[curBlock] = blockCodeTemp;
			blockCodeTemp.insts.clear();
			curBlock++;
		}
	}

	// find MLP/ILP
	for( auto &block : blockCode )
	{
		cout << "Block " << dec << block.first << ":" << endl;

		vector<MLP_entry> mlp_vec;
		vector<string> dest_regs;
		int block_inst_idx = 0;

		// first instruction starts a group
		block.second.group_starts.push_back(0);

		for( auto inst : block.second.insts )
		{
			if( inst.opcode == Load && (inst.srcs[0].type == Memory || inst.srcs[0].type == Cache) )
			{
				// any previous entry not yet used gains a level of parallelism
				for( int i = 0; i < mlp_vec.size(); i++  )
				{
					if( mlp_vec[i].use_idx == -1 )
					{
						mlp_vec[i].load_count++;
					}
				}

				// add an entry for this load instruction
				mlp_vec.push_back( MLP_entry( inst.dest.location, block_inst_idx ) );
			}

			bool found_dep = false;
			if( block.second.insts[block_inst_idx-1].opcode == Barrier )
			{
				// end of group
				found_dep = true;
			}
			else
			{
				for( auto src : inst.srcs )
				{
					for( int i = 0; i < mlp_vec.size(); i++  )
					{
						// check if this src is the result of a memory access
						if( src.location == mlp_vec[i].reg )
						{
							mlp_vec[i].use_idx = block_inst_idx;
						}
					}

					if( !found_dep )
					{
						vector<string> src_strings = extract_srcs( src.location );

						for( auto src_str : src_strings )
						{
							if( find(dest_regs.begin(), dest_regs.end(), src_str ) != dest_regs.end() )
							{
								// data dependency
								block.second.group_starts.push_back(block_inst_idx);

								found_dep = true;
								break;
							}

							if( found_dep ) break;
						}
					}
				}
			}

			// if we found a dependency on this instruction, clear out destination registers
			if( found_dep ) 
			{
				dest_regs.clear();
				cout << "\tEnd of group" << endl;
			}

			// add destination to destinations, this needs to be after 
			// handling all the source registers
			if( inst.dest.location != "" )
				dest_regs.push_back(inst.dest.location);

			cout << inst.toString() << endl;
			block_inst_idx++;
		}

		double mlp_sum = 0.0;
		for( auto entry: mlp_vec )
		{
			mlp_sum += entry.load_count;
		}

		// calculate ILP/MLP
		block.second.ILP = (double)block.second.insts.size() / (double)block.second.group_starts.size();

		if( mlp_vec.size() > 0 )
		{
			block.second.MLP = mlp_sum / (double)mlp_vec.size();
		}
		else
		{
			block.second.MLP = 0.0;
		}

		cout << "ILP = " << block.second.ILP << endl;
		cout << "MLP = " << block.second.MLP << endl;
	}

	// find flow through basic blocks
	for( auto &blockPair : blockCode )
	{
		// is last instruction a branch?
		if( blockPair.second.insts[blockPair.second.insts.size()-1].opcode == Branch ||
				blockPair.second.insts[blockPair.second.insts.size()-1].opcode == Exit )
		{
			
			if( blockPair.second.insts[blockPair.second.insts.size()-1].opcode == Branch )
			{
				// add the block that is at this address
				stringstream ss( blockPair.second.insts[blockPair.second.insts.size()-1].dest.location.substr(2) );
				int addr;
				ss >> hex >> addr;

				// find which block this address is
				for( int i = 1; i < startOfBlocks.size(); i++ )
				{
					if( startOfBlocks[i] == addr )
					{
						blockPair.second.next_blocks.push_back( i );
						break;
					}
				}
			}
			else
			{
				// exit
				blockPair.second.next_blocks.push_back( -1 );
			}

			// only a condition branch in the block, could also fall through
			if( blockPair.second.insts.size() == 1 && blockPair.second.insts[0].cond.isCondition )
			{
				blockPair.second.next_blocks.push_back( blockPair.first + 1 );
			}
		}
		else
		{
			// not ending in a branch, so we fall through
			
			// are we unconditional execution?
			if( !blockPair.second.insts[0].cond.isCondition )
			{
				std::vector<std::string> cond_regs;

				// also fall through to second block if next is conditional, if there is a block after next
				for( int i = blockPair.first+1; i < blockCode.size(); i++ )
				{
					if( blockCode[i].insts[0].cond.isCondition )
					{
						blockPair.second.next_blocks.push_back( i );

						// have we seen the inverse of this condition yet?
						std::string inv_reg = blockCode[i].insts[0].cond.reg;

						if( inv_reg[0] == '!' ) inv_reg = inv_reg.substr(1);
						else 					inv_reg = "!"+inv_reg;

						if( find( cond_regs.begin(), cond_regs.end(), inv_reg ) != cond_regs.end() )
						{
							// we have, can't fall through A and !A, so stop here
							break;
						}
						cond_regs.push_back( blockCode[i].insts[0].cond.reg );
					}
					else
					{
						// no longer a conditional, can't fall though anymore
						blockPair.second.next_blocks.push_back( i );
						break;
					}
				}
			}
			else
			{
				std::vector<std::string> cond_regs;

				// we are a conditional execution, fall through but not to our inverse
				for( int i = blockPair.first + 1; i < blockCode.size(); i++ )
				{
					// is this block a different conditional
					if( blockCode[i].insts[0].cond.isCondition )
					{
						// different PX registers, not our inverse
						if( blockCode[i].insts[0].cond.reg[blockCode[i].insts[0].cond.reg.size()-1] != 
								blockPair.second.insts[0].cond.reg[blockPair.second.insts[0].cond.reg.size()-1] )
						{
							blockPair.second.next_blocks.push_back( i );
						}

						// is this the same conditional
						if( blockCode[i].insts[0].cond.reg == blockPair.second.insts[0].cond.reg )
						{
							// stop here, we can't fall through our conditional
							blockPair.second.next_blocks.push_back( i );
							break;
						}


						// have we seen the inverse of this condition yet?
						std::string inv_reg = blockCode[i].insts[0].cond.reg;

						if( inv_reg[0] == '!' ) inv_reg = inv_reg.substr(1);
						else 					inv_reg = "!"+inv_reg;

						if( find( cond_regs.begin(), cond_regs.end(), inv_reg ) != cond_regs.end() )
						{
							// we have, can't fall through A and !A, so stop here
							break;
						}
					}
					else
					{
						// not a conditional, can't fall through anymore
						blockPair.second.next_blocks.push_back( i );
						break;
					}
				}
			}
		}

#ifdef DEBUG
		cout << "Block " << blockPair.first << " has " << blockPair.second.next_blocks.size() << " next blocks" << endl;
		for( int next : blockPair.second.next_blocks )
		{
			cout << "\tBlock " << next << endl;
		}
#endif
	}

	string graphName( argv[1] );

	if( graphName.find('/') != string::npos )
		graphName = graphName.substr(graphName.rfind('/')+1);
	graphName = graphName.substr(0, graphName.find('.'));

	graphName += ".dot";

	output_program_graph( graphName, blockCode );

	return 0;
}

void parse_instruction( instruction &inst, string s )
{
	// parse instruction
	stringstream ss(s);
	vector<string> tokens;
	string token;

	// tokenize
	while( ss >> token )
	{ 
		const string sep = ",;";

		if( sep.find(token[token.size()-1]) != string::npos )
		{
			token = token.substr(0, token.size()-1);
		}
		tokens.push_back(token);
	}

	// reset instruction data
	inst.opcode = opcode_type::Normal;
	inst.dest.location = "";
	inst.dest.type = operand_type::None;
	inst.srcs = vector<operand>();

	// strip conditional execution if present
	if( tokens[0][0] == '@' )
	{
		inst.cond.isCondition = true;
		inst.cond.reg = tokens[0].substr(1);

		tokens.erase(tokens.begin());
	}
	else
	{
		inst.cond.isCondition = false;
		inst.cond.reg = "";
	}

	inst.opcode_str = tokens[0];

	// see if this is an instruction with args
	if( tokens.size() > 1 )
	{
		// find operand type
		// check for instructions without operands
		if( tokens[0].find("BAR") != string::npos )
		{
			inst.opcode = opcode_type::Barrier;
		} 
		else if( tokens[0].find("BRA") != string::npos )
		{
			inst.opcode = opcode_type::Branch;

			inst.dest.type = operand_type::Constant;
			inst.dest.location = tokens[1];
		}
		else
		{

			// instruction with operands, parse out destination
			if( tokens[1][0] == '[' )
			{
				// destination is a memory location
				inst.dest.type = operand_type::Memory;
				inst.opcode = opcode_type::Store;
			}
			else
			{
				// destination is something else(register)
				inst.dest.type = operand_type::Register;
			}
			inst.dest.location = tokens[1];

			// now parse out all sources
			// for each source operand
			for( int i = 2; i < tokens.size(); i++ )
			{
				operand src;

				src.location = "";
				src.type = operand_type::None;

				// check if an access to special memory,
				if( tokens[i].length() == 1 )
				{
					switch( tokens[i][0] )
					{
						case 'c': 
						case 'g': case 's':
							src.location = tokens[i] + tokens[i+1] + tokens[i+2];
							src.type = operand_type::Cache;

							// don't count constant cache as a load
							if( tokens[i][0] == 's' || tokens[i][0] == 'g' )
								inst.opcode = opcode_type::Load;

							i+=2; 
							break;
					}
				}

				// make sure it wasn't a cache access
				if( src.location.size() == 0 )
				{
					if( tokens[i][0] == '[' )
					{
						// source is memory
						src.type = operand_type::Memory;
						inst.opcode = opcode_type::Load;
					}
					else
					{
						// register access
						src.type = operand_type::Register;
					}
					src.location = tokens[i];
				}
				inst.srcs.push_back(src);
			}

			//  is this a conditional statement, don't count it as a load
			if( tokens[0].find("SETP") != string::npos )
			{
				inst.opcode = opcode_type::Setp;
			}
		}
	}
	else	// no args
	{
		if( tokens[0].find("EXIT") != string::npos )
		{
			inst.opcode = opcode_type::Exit;
		}
	}
}

vector<string> extract_srcs( string src )
{
	vector<string> deps;

	if( src[0] == 'c' || src[0] == 'g' )
	{
		deps.push_back(src);

		// if cache access
		src = src.substr(2);

		int mid = src.find("][");

		// process each inner address separately
		vector<string> a = extract_srcs(src.substr(0, mid));
		deps.insert(deps.end(), a.begin(), a.end() );

		a = extract_srcs(src.substr(mid+2));
		deps.insert(deps.end(), a.begin(), a.end() );
	}
	else if( src[0] == '[' )
	{
		// memory access
		deps.push_back(src);

		vector<string> a = extract_srcs( src.substr(1, src.size()-3) );
		deps.insert(deps.begin(), a.begin(), a.end());
	}
	else
	{
		// extract just the register names
		if( src.find("+") != string::npos )
		{
			int mid = src.find("+");

			// process each half
			vector<string> a = extract_srcs(src.substr(0, mid));
			deps.insert(deps.end(), a.begin(), a.end() );

			a = extract_srcs(src.substr(mid+1));
			deps.insert(deps.end(), a.begin(), a.end() );
		}
		else if( src.find("-") != string::npos )
		{
			int mid = src.find("-");

			// process each half
			vector<string> a = extract_srcs(src.substr(0, mid));
			deps.insert(deps.end(), a.begin(), a.end() );

			a = extract_srcs(src.substr(mid+1));
			deps.insert(deps.end(), a.begin(), a.end() );
		}
		else
		{
			// not compound
			if( src.find("R") != string::npos )
			{
				// register, add it
				deps.push_back(src);
			}
		}
	}

	return deps;
}

void output_program_graph( string fileName, std::map<int, basic_block> &program )
{
	ofstream outFile;

	cout << "Outputting program graph to " << fileName << endl;
	outFile.open( fileName, fstream::out );

	// header info
	outFile << "digraph {" << endl;

	outFile << "start [shape=Mdiamond,label=Start];" << endl;
	outFile << "finish [shape=Msquare,label=Finish];" << endl;

	// output nodes in graph
	for( auto blockPair : program )
	{
		// check for barrier
		bool barrier = false;
		for( auto inst : blockPair.second.insts )
		{
			if( inst.opcode == Barrier )
			{
				barrier = true;
				break;
			}
		}

		// block Name
		outFile << "bb_" << blockPair.first << " [shape=record,style=filled,fillcolor=\"" << (barrier ? "#0000FF" : "#FFFFFF") << "\",label=\"{Block " << blockPair.first;

		// # insts
		outFile << " | Instructions: " << blockPair.second.insts.size();
		// MLP
		outFile << " | MLP: " << blockPair.second.MLP;
		// ILP
		outFile << " | ILP: " << blockPair.second.ILP;
		// Barrier present
		if( barrier )
		{
			outFile << " | Barrier: True";
		}
		// Conditional?
		if( blockPair.second.insts[0].cond.isCondition )
		{
			outFile << "| Conditional: " << blockPair.second.insts[0].cond.reg;
		}

		
		outFile << "}\"];" << endl;
	}

	outFile << endl << endl;

	outFile << "start->bb_0" << endl;

	// output edges
	for( auto blockPair : program )
	{
		for( int next : blockPair.second.next_blocks )
		{
			if( next >= 0 )
			{
				outFile << "bb_" << blockPair.first << "->bb_" << next << endl;
			}
			else
			{
				outFile << "bb_" << blockPair.first << "->finish" << endl;
			}
		}
	}

	// footer
	outFile << "}" << endl;

	outFile.flush();
	outFile.close();
}
