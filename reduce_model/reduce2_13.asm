	code for sm_13
		Function : _Z7reduce2IdEvPT_S1_j
	/*0000*/     I2I.U32.U16 R1, g [0x1].U16;
	/*0008*/     I2I.U32.U16 R4, g [0x6].U16;
	/*0010*/     I2I.U32.U16 R5, R0L;
	/*0018*/     IMUL32.U16.U16 R0, R4L, R1L;
	/*001c*/     IADD32 R0, R5, R0;
	/*0020*/     ISET.C0 o [0x7f], g [0x6], R0, LE;
	/*0028*/     SHL R0 (C0.EQU), R0, 0x3;
	/*0030*/     IADD R0 (C0.EQU), g [0x4], R0;
	/*0038*/     GLD.S64 R2 (C0.EQU), global14 [R0];
	/*0040*/     MVC R2 (C0.NE), c [0x1] [0x0];
	/*0048*/     MVC R3 (C0.NE), c [0x1] [0x1];
	/*0050*/     R2A A1, R5, 0x3;
	/*0058*/     R2G.U32.U32 g [A1+0x8], R2;
	/*0060*/     R2G.U32.U32 g [A1+0x9], R3;
	/*0068*/     BAR.ARV.WAIT b0, 0xfff;
	/*0070*/     SHR.C0 R6, R1, 0x1;
	/*0078*/     SSY 0xf8;
	/*0080*/     BRA C0.EQ, 0xf8;
	/*0088*/     ISET.C0 o [0x7f], R6, R5, LE;
	/*0090*/     SSY 0xd8;
	/*0098*/     BRA C0.NE, 0xd8;
	/*00a0*/     IADD R0, R6, R5;
	/*00a8*/     R2A A2, R0, 0x3;
	/*00b0*/     MOV32 R0, g [A1+0x8];
	/*00b4*/     MOV32 R1, g [A1+0x9];
	/*00b8*/     MOV32 R2, g [A2+0x8];
	/*00bc*/     MOV32 R3, g [A2+0x9];
	/*00c0*/     DADD R0, R0, R2;
	/*00c8*/     R2G.U32.U32 g [A1+0x8], R0;
	/*00d0*/     R2G.U32.U32 g [A1+0x9], R1;
	/*00d8*/     NOP.S;
	/*00e0*/     BAR.ARV.WAIT b0, 0xfff;
	/*00e8*/     SHR.C0 R6, R6, 0x1;
	/*00f0*/     BRA C0.NE, 0x88;
	/*00f8*/     ISET.S.C0 o [0x7f], R5, R124, NE;
	/*0100*/     RET C0.NE;
	/*0108*/     SHL R2, R4, 0x3;
	/*0110*/     MOV32 R0, g [0x8];
	/*0114*/     MOV32 R1, g [0x9];
	/*0118*/     IADD R2, g [0x5], R2;
	/*0120*/     GST.S64 global14 [R2], R0;
