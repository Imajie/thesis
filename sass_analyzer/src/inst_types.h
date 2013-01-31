/*
 * File:	inst_types.h
 * Author:	James Letendre
 *
 * Types for SASS instructions
 */
#ifndef INST_TYPES_H
#define INST_TYPES_H

#include <string>
#include <vector>
#include <sstream>

enum opcode_type
{
	Normal,
	SFU,
	Store,
	Load,
	Setp,
	Branch,
	Barrier,
	Exit
};

enum operand_type
{
	None,
	Constant,
	Register,
	Memory,
	Cache
};

std::string opcode_toString(opcode_type op)
{
	switch(op)
	{
		case Normal:
			return "NORM";
			break;
		case SFU:
			return "SFU";
			break;
		case Store:
			return "STORE";
			break;
		case Load:
			return "LOAD";
			break;
		case Setp:
			return "SETP";
			break;
		case Branch:
			return "BRA";
			break;
		case Barrier:
			return "BAR";
			break;
		case Exit:
			return "EXIT";
			break;
		default:
			return "UNKNOWN";
			break;
	}
}

typedef struct 
{
	operand_type type;
	std::string location;

	std::string toString()
	{
		return location;
	}
} operand;

typedef struct
{
	bool isCondition;
	std::string reg;

	std::string toString()
	{
		if( isCondition )
		{
			return reg + " ";
		}
		return "";
	}
} condition;

typedef struct
{
	opcode_type opcode;
	std::string opcode_str;
	operand dest;
	std::vector<operand> srcs;
	condition cond;
	uint32_t addr;

	std::string toString()
	{
		std::stringstream temp;
		temp << "\t" << std::hex << addr << "\t" << cond.toString() << opcode_str << "(" << opcode_toString(opcode) << ")" << " " << dest.toString() << " ";

		for( operand src : srcs )
		{
			temp << src.toString() << " ";
		}
		return temp.str();
	}
} instruction;

typedef struct
{
	std::vector<instruction> insts;
	std::vector<int> group_starts;
	std::vector<int> next_blocks;
	double ILP, MLP;
} basic_block;

#endif
