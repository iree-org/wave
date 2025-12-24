
/tmp/tmpngifp_st/isolated_benchmark.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <gemm_kernel>:
	s_load_dwordx2 s[4:5], s[0:1], 0x0                         // 000000001600: C0060100 00000000
	s_load_dwordx2 s[8:9], s[0:1], 0x8                         // 000000001608: C0060200 00000008
	s_load_dwordx2 s[12:13], s[0:1], 0x10                      // 000000001610: C0060300 00000010
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF8CC07F
	s_mov_b32 s6, 0x7ffffffc                                   // 00000000161C: BE8600FF 7FFFFFFC
	s_mov_b32 s7, 0x20000                                      // 000000001624: BE8700FF 00020000
	s_mov_b32 s10, 0x7ffffffc                                  // 00000000162C: BE8A00FF 7FFFFFFC
	s_mov_b32 s11, 0x20000                                     // 000000001634: BE8B00FF 00020000
	s_mov_b32 s14, 0x40000                                     // 00000000163C: BE8E00FF 00040000
	s_mov_b32 s15, 0x20000                                     // 000000001644: BE8F00FF 00020000
	s_mov_b32 s16, s4                                          // 00000000164C: BE900004
	s_and_b32 s17, s5, 0xffff                                  // 000000001650: 8611FF05 0000FFFF
	s_or_b32 s17, s17, 4.0                                     // 000000001658: 8711F611
	s_mov_b32 s18, 0x7ffffffd                                  // 00000000165C: BE9200FF 7FFFFFFD
	s_mov_b32 s19, 0x27000                                     // 000000001664: BE9300FF 00027000
	s_mov_b32 s4, s8                                           // 00000000166C: BE840008
	s_and_b32 s5, s9, 0xffff                                   // 000000001670: 8605FF09 0000FFFF
	s_or_b32 s5, s5, 4.0                                       // 000000001678: 8705F605
	s_mov_b32 s6, 0x7ffffffd                                   // 00000000167C: BE8600FF 7FFFFFFD
	s_mov_b32 s7, 0x27000                                      // 000000001684: BE8700FF 00027000
	s_mov_b32 s24, 0                                           // 00000000168C: BE980080
	s_mov_b32 s25, 1                                           // 000000001690: BE990081
	s_mov_b32 s26, 2                                           // 000000001694: BE9A0082
	v_mov_b32_e32 v4, 0                                        // 000000001698: 7E080280
	v_mov_b32_e32 v5, 0                                        // 00000000169C: 7E0A0280
	v_mov_b32_e32 v6, 0                                        // 0000000016A0: 7E0C0280
	v_mov_b32_e32 v7, 0                                        // 0000000016A4: 7E0E0280

00000000000016a8 <loop_0_header>:
	s_cmp_lt_u32 s24, s26                                      // 0000000016A8: BF0A1A18
	s_cbranch_scc1 loop_0_body                                 // 0000000016AC: BF850001
	s_branch loop_0_exit                                       // 0000000016B0: BF82006D

00000000000016b4 <loop_0_body>:
	s_waitcnt vmcnt(0)                                         // 0000000016B4: BF8C0F70
	s_waitcnt lgkmcnt(0)                                       // 0000000016B8: BF8CC07F
	s_barrier                                                  // 0000000016BC: BF8A0000
	v_bfe_u32 v1, v0, 10, 10                                   // 0000000016C0: D1C80001 02291500
	v_bfe_u32 v2, v0, 0, 10                                    // 0000000016C8: D1C80002 02290100
	v_lshrrev_b32_e32 v3, 3, v2                                // 0000000016D0: 20060483
	v_lshl_add_u32 v8, v1, 4, v3                               // 0000000016D4: D1FD0008 040D0901
	v_lshrrev_b32_e32 v9, 5, v8                                // 0000000016DC: 20121085
	v_mov_b32_e32 v8, 0xffffe000                               // 0000000016E0: 7E1002FF FFFFE000
	v_mul_lo_u32 v10, v9, v8                                   // 0000000016E8: D285000A 00021109
	v_lshlrev_b32_e32 v8, 4, v2                                // 0000000016F0: 24100484
	v_add_u32_e32 v9, v10, v8                                  // 0000000016F4: 6812110A
	v_mov_b32_e32 v11, s24                                     // 0000000016F8: 7E160218
	v_lshlrev_b32_e32 v12, 7, v11                              // 0000000016FC: 24181687
	v_add_u32_e32 v11, v9, v12                                 // 000000001700: 68161909
	v_lshlrev_b32_e32 v9, 7, v3                                // 000000001704: 24120687
	v_add_u32_e32 v3, v11, v9                                  // 000000001708: 6806130B
	v_lshlrev_b32_e32 v11, 12, v1                              // 00000000170C: 2416028C
	v_add_u32_e32 v13, v3, v11                                 // 000000001710: 681A1703
	v_mov_b32_e32 v3, s2                                       // 000000001714: 7E060202
	v_lshl_add_u32 v14, v3, 13, v13                            // 000000001718: D1FD000E 04351B03
	v_mov_b32_e32 v13, 0x1000                                  // 000000001720: 7E1A02FF 00001000
	v_lshrrev_b32_e32 v15, 6, v2                               // 000000001728: 201E0486
	v_lshl_add_u32 v16, v1, 1, v15                             // 00000000172C: D1FD0010 043D0301
	v_lshrrev_b32_e32 v17, 2, v16                              // 000000001734: 20222082
	v_mov_b32_e32 v16, 0xfffff000                              // 000000001738: 7E2002FF FFFFF000
	v_mul_lo_u32 v18, v17, v16                                 // 000000001740: D2850012 00022111
	v_add_u32_e32 v16, v13, v18                                // 000000001748: 6820250D
	v_lshlrev_b32_e32 v13, 10, v15                             // 00000000174C: 241A1E8A
	v_add_u32_e32 v17, v16, v13                                // 000000001750: 68221B10
	v_lshlrev_b32_e32 v16, 11, v1                              // 000000001754: 2420028B
	v_add_u32_e32 v18, v17, v16                                // 000000001758: 68242111
	s_nop 0                                                    // 00000000175C: BF800000
	v_readfirstlane_b32 s8, v18                                // 000000001760: 7E100512
	s_mov_b32 m0, s8                                           // 000000001764: BEFC0008
	buffer_load_dwordx4 v14, s[16:19], 0 offen lds             // 000000001768: E05D1000 8004000E
	v_add_u32_e32 v14, v10, v8                                 // 000000001770: 681C110A
	v_add_u32_e32 v8, v14, v12                                 // 000000001774: 6810190E
	v_add_u32_e32 v10, v8, v9                                  // 000000001778: 68141308
	v_add_u32_e32 v8, v10, v11                                 // 00000000177C: 6810170A
	v_mov_b32_e32 v9, s3                                       // 000000001780: 7E120203
	v_lshl_add_u32 v10, v9, 13, v8                             // 000000001784: D1FD000A 04211B09
	v_add_u32_e32 v8, v13, v16                                 // 00000000178C: 6810210D
	v_lshrrev_b32_e32 v11, 1, v1                               // 000000001790: 20160281
	v_lshrrev_b32_e32 v12, 2, v15                              // 000000001794: 20181E82
	v_add_u32_e32 v13, v11, v12                                // 000000001798: 681A190B
	v_and_b32_e32 v11, 0, v13                                  // 00000000179C: 26161A80
	v_lshl_add_u32 v12, v11, 12, v8                            // 0000000017A0: D1FD000C 0421190B
	s_nop 0                                                    // 0000000017A8: BF800000
	v_readfirstlane_b32 s8, v12                                // 0000000017AC: 7E10050C
	s_mov_b32 m0, s8                                           // 0000000017B0: BEFC0008
	buffer_load_dwordx4 v10, s[4:7], 0 offen lds               // 0000000017B4: E05D1000 8001000A
	s_waitcnt vmcnt(0)                                         // 0000000017BC: BF8C0F70
	s_waitcnt lgkmcnt(0)                                       // 0000000017C0: BF8CC07F
	s_barrier                                                  // 0000000017C4: BF8A0000
	v_and_b32_e32 v8, 63, v2                                   // 0000000017C8: 261004BF
	v_lshrrev_b32_e32 v10, 4, v8                               // 0000000017CC: 20141084
	v_lshlrev_b32_e32 v8, 3, v10                               // 0000000017D0: 24101483
	v_and_b32_e32 v11, 15, v2                                  // 0000000017D4: 2616048F
	v_lshlrev_b32_e32 v2, 7, v11                               // 0000000017D8: 24041687
	v_or_b32_e32 v12, v8, v2                                   // 0000000017DC: 28180508
	v_or_b32_e32 v13, v12, v16                                 // 0000000017E0: 281A210C
	ds_read_b64 v[16:17], v13                                  // 0000000017E4: D8EC0000 1000000D
	ds_read_b64 v[18:19], v13 offset:32                        // 0000000017EC: D8EC0020 1200000D
	ds_read_b64 v[20:21], v13 offset:64                        // 0000000017F4: D8EC0040 1400000D
	ds_read_b64 v[22:23], v13 offset:96                        // 0000000017FC: D8EC0060 1600000D
	v_or_b32_e32 v12, v8, v2                                   // 000000001804: 28180508
	v_lshlrev_b32_e32 v2, 11, v15                              // 000000001808: 24041E8B
	v_or_b32_e32 v8, v12, v2                                   // 00000000180C: 2810050C
	ds_read_b64 v[12:13], v8 offset:4096                       // 000000001810: D8EC1000 0C000008
	ds_read_b64 v[24:25], v8 offset:4128                       // 000000001818: D8EC1020 18000008
	ds_read_b64 v[26:27], v8 offset:4160                       // 000000001820: D8EC1040 1A000008
	ds_read_b64 v[28:29], v8 offset:4192                       // 000000001828: D8EC1060 1C000008
	s_waitcnt lgkmcnt(0)                                       // 000000001830: BF8CC07F
	v_mfma_f32_16x16x16_f16 v[4:7], v[12:13], v[16:17], v[4:7] // 000000001834: D3CD0004 0412210C
	s_waitcnt lgkmcnt(0)                                       // 00000000183C: BF8CC07F
	v_mfma_f32_16x16x16_f16 v[4:7], v[24:25], v[18:19], v[4:7] // 000000001840: D3CD0004 04122518
	s_waitcnt lgkmcnt(0)                                       // 000000001848: BF8CC07F
	v_mfma_f32_16x16x16_f16 v[4:7], v[26:27], v[20:21], v[4:7] // 00000000184C: D3CD0004 0412291A
	s_waitcnt lgkmcnt(0)                                       // 000000001854: BF8CC07F
	v_mfma_f32_16x16x16_f16 v[4:7], v[28:29], v[22:23], v[4:7] // 000000001858: D3CD0004 04122D1C

0000000000001860 <loop_0_latch>:
	s_add_u32 s24, s24, s25                                    // 000000001860: 80181918
	s_branch loop_0_header                                     // 000000001864: BF82FF90

0000000000001868 <loop_0_exit>:
	v_lshlrev_b32_e32 v2, 6, v1                                // 000000001868: 24040286
	v_lshl_or_b32 v1, v11, 2, v2                               // 00000000186C: D2000001 0409050B
	v_lshl_add_u32 v2, v9, 7, v1                               // 000000001874: D1FD0002 04050F09
	v_lshl_add_u32 v1, v10, 12, v2                             // 00000000187C: D1FD0001 0409190A
	v_lshl_add_u32 v2, v15, 14, v1                             // 000000001884: D1FD0002 04051D0F
	v_lshl_add_u32 v1, v3, 15, v2                              // 00000000188C: D1FD0001 04091F03
	s_waitcnt vmcnt(0)                                         // 000000001894: BF8C0F70
	buffer_store_dword v4, v1, s[12:15], 0 offen               // 000000001898: E0701000 80030401
	s_waitcnt vmcnt(0)                                         // 0000000018A0: BF8C0F70
	buffer_store_dword v5, v1, s[12:15], 0 offen offset:1024   // 0000000018A4: E0701400 80030501
	s_waitcnt vmcnt(0)                                         // 0000000018AC: BF8C0F70
	buffer_store_dword v6, v1, s[12:15], 0 offen offset:2048   // 0000000018B0: E0701800 80030601
	s_waitcnt vmcnt(0)                                         // 0000000018B8: BF8C0F70
	buffer_store_dword v7, v1, s[12:15], 0 offen offset:3072   // 0000000018BC: E0701C00 80030701
	s_endpgm                                                   // 0000000018C4: BF810000
