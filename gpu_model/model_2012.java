import java.util.Scanner;

public class model_2012 {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// DEBUG flag
		boolean DEBUG = false;
		if( args.length > 0 ) DEBUG = true;

		//Top level variables
		double T_exec;				// overall execution time
		double T_comp;				// time to execute compute instrs (& cost to execute memory instrs)
		double T_mem;				// time spent on memory requets and transfers
		double T_overlap;			// amount of memory access cost that can be hidden by computation
		
		//Intermediary variables
		double W_parallel;			// parallelizable base execution time
		double W_serial; 			// overhead costs due to serialization
		double ITILP;				// inter thread instruction level parallelism
		double ITILP_max;			// max ITILP req'd to fully hid pipeline latency
		double O_sync;				// synchronization overhead
		double O_SFU;				// special function unit (SFU) overhead
		double F_sync;				// Fraction of synch cost hidden
		double F_SFU;				// visibility of SFU instructions
		double AMAT;				// average memory access latency (including cache effect)
		double ITMLP;				// inter thread memory level parallelism
		double MWP_cp;				// number of warps who memory req's are overlapped during one computation period
		double F_overlap;			// approximate overlap of T_comp & T_mem
		double zeta;				// used for F_overlap computation
		double CWP;					// computation warp parallelism
		double CWP_full;			// when there is enough number of warps
		double comp_cycles; 		// computation cycles per warp
		double mem_cycles;			// memory waiting cycles per warp
		double MWP;					// memory warp parallelism
		double MWP_peak_bw;			// number of memory warps per SM under peak memory bandwidth
		double MWP_without_bw_full;	// max MWP without limiting for bandwidth
		double BW_per_warp;			// bandwidth requirement per warp
		double mem_lat_final;		// Effective memory latency for computations

		double total_trans;			// Total memory transactions
		double num_trans_per_req;	// Number of memory transactions per memory request
		double weight_32;			// Weights for memory transactions of each size
		double weight_64;
		double weight_128;
		double avg_dep_delay_per_trans;	// Average departure delay per memory transaction
		double data_per_req;		// average data loaded per memory request
		double dep_delay_per_req;	// departure delay for each memory request
		double N_active_warps_MWP;	// Effective number of active warps after effects of memory merging
		
		// Card Specific Variables
		//------------------------------------------------------------
		double freq 				= 1.15;			// GPU core clock frequency in GHz
		double mem_peak_bandwidth	= 144;			// GPU memory bandwidth in GBps
		double N_SM					= 14;			// The number of SMs
		double SIMD_width			= 32;			// The number of SPs per SM
		double warp_size			= 32;			// equal to SIMD_width
		double SFU_width			= 4;			// The number of SFUs in each SM
		double transaction_size		= 128;			// transaction size for a DRAM request in Bytes
		double DRAM_lat				= 440;			// baseline DRAM access latency
		double delta				= 20;			// transaction departure delay
		double l1_lat				= 18;			// latency of L1 Cache
		double l2_lat				= 130;			// latency of L1 Cache
		double max_warps			= 48;			// Maximum number of warps per SM
		double avg_inst_lat			= 18;			// approximated as average FP ops, paper has FP_lat = 18, thesis says most instructions take 4 cycles
		double gamma				= 64;			// machine dependent parameter
		double delay_32				= 37;			// departure delays for memory transactions of each size, from thesis
		double delay_64				= 37;
		double delay_128			= 58;

		// Read parameters---------------------------------------------------------------------
		Scanner in = new Scanner(System.in);

		// Kernel launch params
		//System.out.print("Threads = ");
		double threads 			= in.nextInt();					// Threads in a block
		//System.out.print("Blocks = ");
		double blocks			= in.nextInt();					// Blocks to execute in the grid
		double N_total_warps	= (threads/warp_size)*blocks;	// total warps executed in the kernel

		// Instruction counts
		//System.out.print("Mem_insts = ");
		double N_mem_insts		= in.nextInt();
		//System.out.print("Sync_insts = ");
		double N_sync_insts		= in.nextInt();
		//System.out.print("SFU_insts = ");
		double N_SFU_insts		= in.nextInt();
		//System.out.print("total_insts = ");
		double N_insts			= in.nextInt();

		// Occupancy
		//System.out.print("active warps = ");
		double N_active_warps	= in.nextInt();
		//System.out.print("active SMs = ");
		double N_active_SMs		= in.nextInt();

		// Scale instruction counts so they are number per warp
		//N_mem_insts 			/= N_active_warps;
		N_mem_insts				/= N_total_warps / (3*N_active_SMs);

		//N_sync_insts			/= N_active_warps;
		//N_sync_insts			/= N_total_warps;
		
		//N_SFU_insts 			/= N_active_warps;
		N_SFU_insts				/= N_total_warps / N_active_SMs;

		//N_insts 				/= N_active_warps;
		N_insts 				/= N_total_warps / N_active_SMs;

		// Additional overhead
		//System.out.print("CF overhead = ");
		double O_CFdiv			= in.nextDouble();
		//System.out.print("bank overhead = ");
		double O_bank			= in.nextDouble();

		// Memory accesses
		//System.out.print("total memory requests = ");
		double total_req		= in.nextInt() / N_active_warps;
		//System.out.print("32 bit transactions = ");
		double trans_32			= in.nextInt() / (3*N_active_warps);
		//System.out.print("64 bit transactions = ");
		double trans_64			= in.nextInt() / (3*N_active_warps);
		//System.out.print("128 bit transactions = ");
		double trans_128		= in.nextInt() / (3*N_active_warps);
		//System.out.print("independent loads = ");
		double independent_loads = in.nextDouble();
		//System.out.print("duplicate loads = ");
		double duplicate_loads	= in.nextDouble();

		// Cache
		//System.out.print("L1 miss = ");
		double l1_miss_ratio	= in.nextDouble();
		//System.out.print("L2 miss = ");
		double l2_miss_ratio	= in.nextDouble();
		double miss_ratio		= l1_miss_ratio*l2_miss_ratio;			// global memory cache (combined L1 & L2) miss ratio
		double hit_lat 			= l1_lat + l1_miss_ratio*l2_lat;		// global memory cache (combined L1 & L2) hit latency

		// Kernel parallelism
		//System.out.print("ILP = ");
		double ILP				= in.nextDouble();
		//System.out.print("MLP = ");
		double MLP				= in.nextDouble();

		System.out.println();
		// Common calculations-----------------------------------------------------------------
		// ITILP
		ITILP_max = avg_inst_lat/(warp_size/SIMD_width);	// Equation (5)
		ITILP = Math.min(ILP * N_active_warps, ITILP_max);	// Equation (4)
		
		// MWP common
		total_trans = trans_32 + trans_64 + trans_128;
		num_trans_per_req = total_trans/total_req;

		weight_32  = trans_32/total_trans;
		weight_64  = trans_64/total_trans;
		weight_128 = trans_128/total_trans;

		avg_dep_delay_per_trans = delay_32*weight_32 + delay_64*weight_64 + delay_128*weight_128;

		mem_lat_final = DRAM_lat + (num_trans_per_req - 1)*avg_dep_delay_per_trans;

		// MWP with BW limit
		data_per_req = (trans_32*32 + trans_64*64 + trans_128*128)/(total_req);
		BW_per_warp = (freq * data_per_req) / mem_lat_final;						// Equation (A.7)
		MWP_peak_bw = mem_peak_bandwidth / (BW_per_warp * N_active_SMs);			// Equation (A.6)

		// MWP without BW limit
		dep_delay_per_req	= num_trans_per_req * avg_dep_delay_per_trans;
		MWP_without_bw_full = mem_lat_final / dep_delay_per_req;

		// MWP active warp limit
		// sanity check
		if( independent_loads < 1 ) independent_loads = 1; 
		if( duplicate_loads < 1 )   duplicate_loads = 1;
		N_active_warps_MWP = Math.ceil(N_active_warps * ( independent_loads/duplicate_loads ) );
		
		// MWP final
		MWP = Math.min(Math.min( MWP_without_bw_full , MWP_peak_bw), N_active_warps);	// Equation (A.5)

		// Memory access time
		AMAT = mem_lat_final * miss_ratio + hit_lat;							// Equation (12)
		
		// CWP
		mem_cycles = (N_mem_insts * AMAT) / MLP;			// Equation (A.4)
		comp_cycles = (N_insts * avg_inst_lat) / ITILP;		// Equation (A.3)
		CWP_full = (mem_cycles + comp_cycles)/comp_cycles;	// Equation (A.2)
		CWP = Math.min(CWP_full, N_active_warps);			// Equation (A.1)

		// ITLMP
		MWP_cp = Math.min(Math.max(1, (CWP - 1)), MWP);		// Equation (15)
		ITMLP = Math.min( (MLP * MWP_cp) , MWP_peak_bw);	// Equation (14)
		
		// T_comp------------------------------------------------------------------------------
		// Sync overhead
		F_sync = gamma * mem_lat_final * (N_mem_insts / N_insts);		// Equation (8)
		O_sync = ((N_sync_insts * blocks)/N_active_SMs) * F_sync;		// Equation (7)	XXX
		
		// SFU overhead
		F_SFU = Math.min(Math.max( ((N_SFU_insts/N_insts) - (SFU_width/SIMD_width)) , 0), 1);	// Equation (10)
		O_SFU = ((N_SFU_insts * N_total_warps)/N_active_SMs) * (warp_size / SFU_width) * F_SFU;	// Equation (9)
		
		// Serial and Parallel time
		W_serial = O_sync + O_SFU + O_CFdiv + O_bank;									// Equation (6)
		W_parallel = ((N_insts * N_total_warps)/N_active_SMs) * (avg_inst_lat/ITILP);	// Equation (3)
		
		T_comp = W_parallel + W_serial;		// Equation (2)

		// T_mem-------------------------------------------------------------------------------
		T_mem = (N_mem_insts * N_total_warps) / (N_active_SMs * ITMLP) * AMAT;	// Equation (11)

		// T_overlap---------------------------------------------------------------------------
		zeta = CWP <= MWP ? 1 : 0;								// Equation (17)
		F_overlap = (N_active_warps - zeta) / N_active_warps;	// Equation (17)
		T_overlap = Math.min((T_comp * F_overlap), T_mem);		// Equation (16)

		// T_exec------------------------------------------------------------------------------
		T_exec = T_comp + T_mem - T_overlap;	// Equation (1)

		if( DEBUG )
		{
			System.out.println("==============DEBUG==============");
			System.out.println("==============Card Params========");
			System.out.println("avg_inst_lat\t= " + avg_inst_lat);
			System.out.println("freq\t\t= " + freq);
			System.out.println("mem_peak_bw\t= " + mem_peak_bandwidth);
			System.out.println("N_SM\t\t= " + N_SM);
			System.out.println("SIMD_width\t= " + SIMD_width);
			System.out.println("warp_size\t= " + warp_size);
			System.out.println("SFU_width\t= " + SFU_width);
			System.out.println("DRAM_lat\t= " + DRAM_lat);
			System.out.println("delta\t\t= " + delta);
			System.out.println("mem_tran_size\t= " + transaction_size);
			System.out.println("l1_lat\t\t= " + l1_lat);
			System.out.println("l2_lat\t\t= " + l2_lat);
			System.out.println("max_warps\t= " + max_warps);
			System.out.println("gamma\t\t= " + gamma);
			System.out.println("==============Code Params========");
			System.out.println("threads\t\t= " + threads);
			System.out.println("blocks\t\t= " + blocks);
			System.out.println("N_total_warps\t= " + N_total_warps);
			System.out.println("N_mem_insts\t= " + N_mem_insts);
			System.out.println("N_sync_insts\t= " + N_sync_insts);
			System.out.println("N_SFU_insts\t= " + N_SFU_insts);
			System.out.println("N_insts\t\t= " + N_insts);
			System.out.println("N_active_warps\t= " + N_active_warps);
			System.out.println("N_active_SMs\t= " + N_active_SMs);
			System.out.println("l1_miss_ratio\t= " + l1_miss_ratio);
			System.out.println("l2_miss_ratio\t= " + l2_miss_ratio);
			System.out.println("miss_ratio\t= " + miss_ratio);
			System.out.println("hit_lat\t\t= " + hit_lat);
			System.out.println("ILP\t\t= " + ILP);
			System.out.println("MLP\t\t= " + MLP);
			System.out.println("==============Equations==========");
			System.out.println("T_exec\t\t\t= " + T_exec);
			System.out.println("T_comp\t\t\t= " + T_comp);
			System.out.println("W_parallel\t\t= " + W_parallel);
			System.out.println("ITILP\t\t\t= " + ITILP);
			System.out.println("ITILP_max\t\t= " + ITILP_max);
			System.out.println("W_serial\t\t= " + W_serial);
			System.out.println("O_sync\t\t\t= " + O_sync);
			System.out.println("F_sync\t\t\t= " + F_sync);
			System.out.println("O_SFU\t\t\t= " + O_SFU);
			System.out.println("F_SFU\t\t\t= " + F_SFU);
			System.out.println("O_CFdiv\t\t\t= " + O_CFdiv);
			System.out.println("O_bank\t\t\t= " + O_bank);
			System.out.println("T_mem\t\t\t= " + T_mem);
			System.out.println("AMAT\t\t\t= " + AMAT);
			System.out.println("mem_lat_final\t\t= " + mem_lat_final);
			System.out.println("total_trans\t\t= " + total_trans);
			System.out.println("num_trans_per_req\t= " + num_trans_per_req);
			System.out.println("weight_32\t\t= " + weight_32);
			System.out.println("weight_64\t\t= " + weight_64);
			System.out.println("weight_128\t\t= " + weight_128);
			System.out.println("avg_dep_delay_per_trans\t= " + avg_dep_delay_per_trans);
			System.out.println("data_per_req\t\t= " + data_per_req);
			System.out.println("dep_delay_per_req\t= " + dep_delay_per_req);
			System.out.println("ITMLP\t\t\t= " + ITMLP);
			System.out.println("MWP_cp\t\t\t= " + MWP_cp);
			System.out.println("T_overlap\t\t= " + T_overlap);
			System.out.println("F_overlap\t\t= " + F_overlap);
			System.out.println("zeta\t\t\t= " + zeta);
			System.out.println("CWP\t\t\t= " + CWP);
			System.out.println("CWP_full\t\t= " + CWP_full);
			System.out.println("comp_cycles\t\t= " + comp_cycles);
			System.out.println("mem_cycles\t\t= " + mem_cycles);
			System.out.println("MWP\t\t\t= " + MWP);
			System.out.println("MWP_without_bw_full\t= " + MWP_without_bw_full);
			System.out.println("MWP_peak_bw\t\t= " + MWP_peak_bw);
			System.out.println("N_active_warps_MWP\t= " + N_active_warps_MWP);
			System.out.println("independent_loads\t= " + independent_loads);
			System.out.println("duplicate_loads\t\t= " + duplicate_loads);
			System.out.println("BW_per_warp\t\t= " + BW_per_warp);
			System.out.println("==============END DEBUG==============");
		}

		// Print Results
		System.out.println("Time(cycles)");
		System.out.println("T_mem\t\t= " + Math.ceil(T_mem));
		System.out.println("T_comp\t\t= " + Math.ceil(T_comp));
		System.out.println("T_overlap\t= " + Math.ceil(T_overlap));
		System.out.println("T_exec\t\t= " + Math.ceil(T_exec));
		System.out.println("=====================================");
		System.out.println("Time(usec)");
		System.out.println("T_mem\t\t= " + Math.ceil(T_mem)/(freq*1e3));
		System.out.println("T_comp\t\t= " + Math.ceil(T_comp)/(freq*1e3));
		System.out.println("T_overlap\t= " + Math.ceil(T_overlap)/(freq*1e3));
		System.out.println("T_exec\t\t= " + Math.ceil(T_exec)/(freq*1e3));

	}
}
