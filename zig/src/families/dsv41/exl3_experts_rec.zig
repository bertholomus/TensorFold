//! exl3_experts.decode against the served build's decode launches as the recording saw them on the GPU
//! (zigport/rec/m1e/rank0/ztrace.jsonl: one line per distinct launch, device addresses masked as <ptr>; the lines
//! below are copied from it verbatim): each launch's kernel, grid, block, shared memory, PDL flag and parameter bytes,
//! for windows of 1 to 63 rows of a decoder layer (layers 0 and 1: 385 experts, 7 slots, gate / up and down K2 6..10,
//! the <2, 10> kernels) and of 5 to 20 rows of an MTP layer (129 experts, 4 slots, K2 8: the <8, 8> kernels).
const std = @import("std");
const cuda = @import("cuda");
const xe = @import("exl3_experts.zig");
const weights = @import("weights.zig");

/// mul1, the codebook of every tensor of our checkpoint
const codebook = 2;

/// A recorded window: its rows, the layer's experts, slots and K2 (min, max) of gate / up and of down (the
/// checkpoint's), and its three launches as recorded (decode_prep, gate / up, down), a line each.
const Case = struct { r: usize, e: u32, slots: usize, k2: [2]u32, rec: []const u8 };

const cases = [_]Case{
    .{ .r = 1, .e = 385, .slots = 7, .k2 = .{ 6, 10 }, .rec =
    \\{"phase":"round:r1:b1024","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[71,1,1],"block":[256,1,1],"smem":56,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:07000000|64:81010000|68:01000000|72:<ptr>|80:<ptr>|88:<ptr>|96:01000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"round:r1:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[7,9,8],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:07000000|92:04000000|96:01000000|100:07000000|104:<ptr>8101000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"round:r1:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[7,40,1],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:07000000|92:01000000|96:01000000|100:07000000|104:<ptr>810100000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 2, .e = 385, .slots = 7, .k2 = .{ 6, 10 }, .rec =
    \\{"phase":"round:r2:b1024","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[141,1,1],"block":[256,1,1],"smem":112,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:07000000|64:81010000|68:02000000|72:<ptr>|80:<ptr>|88:<ptr>|96:02000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"round:r2:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[14,9,8],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:0e000000|92:04000000|96:02000000|100:07000000|104:<ptr>8101000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"round:r2:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[14,40,1],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:0e000000|92:01000000|96:02000000|100:07000000|104:<ptr>810100000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 5, .e = 385, .slots = 7, .k2 = .{ 6, 10 }, .rec =
    \\{"phase":"round:r5:b1024","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[351,1,1],"block":[256,1,1],"smem":280,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:07000000|64:81010000|68:05000000|72:<ptr>|80:<ptr>|88:<ptr>|96:05000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"round:r5:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[35,9,8],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:23000000|92:04000000|96:05000000|100:07000000|104:<ptr>8101000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"round:r5:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[35,40,1],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:23000000|92:01000000|96:05000000|100:07000000|104:<ptr>810100000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 16, .e = 385, .slots = 7, .k2 = .{ 6, 10 }, .rec =
    \\{"phase":"round:r16:b1024","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[1121,1,1],"block":[256,1,1],"smem":896,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:07000000|64:81010000|68:10000000|72:<ptr>|80:<ptr>|88:<ptr>|96:10000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"round:r16:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[112,9,8],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:70000000|92:04000000|96:10000000|100:07000000|104:<ptr>8101000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"round:r16:b1024","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[112,40,1],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:70000000|92:01000000|96:10000000|100:07000000|104:<ptr>810100000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 17, .e = 385, .slots = 7, .k2 = .{ 6, 10 }, .rec =
    \\{"phase":"forward","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[1191,1,1],"block":[256,1,1],"smem":952,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:07000000|64:81010000|68:11000000|72:<ptr>|80:<ptr>|88:<ptr>|96:11000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"forward","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[119,9,16],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:77000000|92:04000000|96:11000000|100:07000000|104:<ptr>8101000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"forward","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[119,40,2],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:77000000|92:01000000|96:11000000|100:07000000|104:<ptr>810100000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 33, .e = 385, .slots = 7, .k2 = .{ 6, 10 }, .rec =
    \\{"phase":"forward","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[2311,1,1],"block":[256,1,1],"smem":1848,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:07000000|64:81010000|68:21000000|72:<ptr>|80:<ptr>|88:<ptr>|96:21000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"forward","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[231,9,24],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:e7000000|92:04000000|96:21000000|100:07000000|104:<ptr>8101000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"forward","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[231,40,3],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:e7000000|92:01000000|96:21000000|100:07000000|104:<ptr>810100000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 63, .e = 385, .slots = 7, .k2 = .{ 6, 10 }, .rec =
    \\{"phase":"forward","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[4411,1,1],"block":[256,1,1],"smem":3528,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:07000000|64:81010000|68:3f000000|72:<ptr>|80:<ptr>|88:<ptr>|96:3f000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"forward","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[385,9,32],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:b9010000|92:04000000|96:3f000000|100:07000000|104:<ptr>8101000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"forward","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi2ELi10ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[385,40,4],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:b9010000|92:01000000|96:3f000000|100:07000000|104:<ptr>810100000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 5, .e = 129, .slots = 4, .k2 = .{ 8, 8 }, .rec =
    \\{"phase":"draft_batch_capture:s1","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[201,1,1],"block":[256,1,1],"smem":160,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:04000000|64:81000000|68:05000000|72:<ptr>|80:<ptr>|88:<ptr>|96:05000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"draft_batch_capture:s1","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[20,9,8],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:14000000|92:04000000|96:05000000|100:04000000|104:<ptr>8100000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"draft_batch_capture:s1","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[20,40,1],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:14000000|92:01000000|96:05000000|100:04000000|104:<ptr>810000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 10, .e = 129, .slots = 4, .k2 = .{ 8, 8 }, .rec =
    \\{"phase":"draft_batch_capture:s2","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[401,1,1],"block":[256,1,1],"smem":320,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:04000000|64:81000000|68:0a000000|72:<ptr>|80:<ptr>|88:<ptr>|96:0a000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"draft_batch_capture:s2","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[40,9,8],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:28000000|92:04000000|96:0a000000|100:04000000|104:<ptr>8100000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"draft_batch_capture:s2","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[40,40,1],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:28000000|92:01000000|96:0a000000|100:04000000|104:<ptr>810000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 15, .e = 129, .slots = 4, .k2 = .{ 8, 8 }, .rec =
    \\{"phase":"draft_batch_capture:s3","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[601,1,1],"block":[256,1,1],"smem":480,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:04000000|64:81000000|68:0f000000|72:<ptr>|80:<ptr>|88:<ptr>|96:0f000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"draft_batch_capture:s3","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[60,9,8],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:3c000000|92:04000000|96:0f000000|100:04000000|104:<ptr>8100000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"draft_batch_capture:s3","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[60,40,1],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:3c000000|92:01000000|96:0f000000|100:04000000|104:<ptr>810000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
    .{ .r = 20, .e = 129, .slots = 4, .k2 = .{ 8, 8 }, .rec =
    \\{"phase":"draft_batch_capture:s4","name":"_ZN43_GLOBAL__N__6130c0d5_10_experts_cu_f13ab8ba18decode_prep_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iiiiPiSB_SB_iPKfSD_SD_PfiiSB_","grid":[801,1,1],"block":[256,1,1],"smem":640,"pdl":0,"via":"drv","params":"0:<ptr>|8:00140000|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:00140000|60:04000000|64:81000000|68:14000000|72:<ptr>|80:<ptr>|88:<ptr>|96:14000000|104:<ptr>|112:<ptr>|120:0000000000000000|128:<ptr>|136:00140000|140:01000000|144:<ptr>"}
    \\{"phase":"draft_batch_capture:s4","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[80,9,16],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:00140000|84:80040000|88:50000000|92:04000000|96:14000000|100:04000000|104:<ptr>8100000000000000<ptr><ptr><ptr><ptr>0000204101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    \\{"phase":"draft_batch_capture:s4","name":"_ZN8tf_exl3x17grouped_cp_kernelILi2ELi8ELi4ELi3ELi8ELi8ELi2EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiNS_9DecodeEpiE","grid":[80,40,2],"block":[128,1,1],"smem":0,"pdl":1,"via":"drv","params":"0:<ptr>|8:<ptr>|16:<ptr>|24:<ptr>|32:<ptr>|40:<ptr>|48:<ptr>|56:<ptr>|64:<ptr>|72:<ptr>|80:80040000|84:00140000|88:50000000|92:01000000|96:14000000|100:04000000|104:<ptr>810000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000<ptr><ptr><ptr>0000000000000000<ptr>0100000000000000<ptr><ptr><ptr><ptr>0100000000000000"}
    },
};

/// Fake device addresses, a distinct one a buffer: 0x7f in bits 40..47, which no other 8-byte word of these launches'
/// parameters has.
fn fake(n: u64) u64 {
    return 0x7f00_0000_0000 + (n << 24);
}

fn isFake(v: u64) bool {
    return v >> 40 == 0x7f;
}

fn experts(c: Case) weights.Experts {
    return .{ .trellis = fake(1), .gate_ptr = fake(2), .up_ptr = fake(3), .down_ptr = fake(4), .gate_k2 = fake(5), .up_k2 = fake(6), .down_k2 = fake(7), .suh_g = fake(8), .suh_u = fake(9), .svh_g = fake(10), .svh_u = fake(11), .suh_d = fake(12), .svh_d = fake(13), .count = c.e, .dims = 5120, .width = 1152, .down_k = 1152, .down_n = 5120, .k2_gu = c.k2, .k2_d = c.k2 };
}

/// The served decode scratch: 64 rows.
fn scratch(slots: usize) xe.DecodeScratch {
    return .{ .rows = 64, .slots = slots, .xg = fake(20), .xu = fake(21), .xd = fake(22), .z = fake(23), .y = fake(24), .cnt_gu = fake(25), .cnt_d = fake(26), .epoch = fake(27), .ready = fake(28), .ready_cnt = fake(29), .ids = fake(30), .count = fake(31), .members = fake(32) };
}

const x = fake(40); // bf16 [R, D], rows contiguous
const pick = fake(41);
const wts = fake(42);
const out = fake(43);
const limit: f32 = 10; // the checkpoint's swiglu_limit, as recorded

/// Parameters as the recorder prints them: "offset:bytes" a parameter, '|' between, the bytes in hex with each 8-byte
/// word that holds a device address as <ptr>.
fn masked(a: *const cuda.Args, buf: []u8) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for (a.offsets[0..a.count], a.sizes[0..a.count], 0..) |off, size, i| {
        if (i > 0) try w.writeByte('|');
        try w.print("{d}:", .{off});
        const b = a.storage[off..][0..size];
        var j: usize = 0;
        while (j < b.len) {
            if ((off + j) % 8 == 0 and j + 8 <= b.len and isFake(std.mem.readInt(u64, b[j..][0..8], .little))) {
                try w.writeAll("<ptr>");
                j += 8;
            } else {
                try w.printHex(b[j..][0..1], .lower);
                j += 1;
            }
        }
    }
    return w.buffered();
}

fn dims(v: std.json.Value) [3]u32 {
    const g = v.array.items;
    return .{ @intCast(g[0].integer), @intCast(g[1].integer), @intCast(g[2].integer) };
}

/// The recorded kernel is the one the launch takes: decode_prep<bf16>, or grouped_cp_kernel<mul1, 8, 4, 3, LO, HI,
/// EPI> with (LO, HI) the launch's K2 range and EPI 1 for gate / up, 2 for down.
fn expectKernel(name: []const u8, l: *const xe.DecodeLaunch) !void {
    if (l.kernel == .prep) return std.testing.expect(std.mem.indexOf(u8, name, "18decode_prep_kernelI13__nv_bfloat16E") != null);
    const at = std.mem.indexOf(u8, name, "17grouped_cp_kernelI") orelse return error.TestUnexpectedResult;
    var t: [7]i64 = undefined;
    try std.testing.expect(xe.templateArgs(name[at + "17grouped_cp_kernel".len ..], &t));
    try std.testing.expectEqualSlices(i64, &.{ codebook, 8, 4, xe.decode_stages }, t[0..4]);
    try std.testing.expectEqual(l.range, xe.range(@intCast(t[4]), @intCast(t[5])));
    try std.testing.expectEqual(@as(i64, if (l.kernel == .gate_up) 1 else 2), t[6]);
}

test "decode's launches are the served build's recorded ones" {
    const gpa = std.testing.allocator;
    for (cases) |c| {
        const ls = try xe.decodeLaunches(experts(c), scratch(c.slots), x, 5120, pick, wts, out, c.r, limit);
        var lines = std.mem.splitScalar(u8, c.rec, '\n');
        for (&ls) |*l| {
            const line = lines.next() orelse return error.TestUnexpectedResult;
            const parsed = try std.json.parseFromSlice(std.json.Value, gpa, line, .{});
            defer parsed.deinit();
            const v = parsed.value.object;
            errdefer std.debug.print("{s}, {d} rows, {s}\n", .{ v.get("phase").?.string, c.r, @tagName(l.kernel) });
            try expectKernel(v.get("name").?.string, l);
            try std.testing.expectEqual(dims(v.get("grid").?), [3]u32{ l.cfg.grid.x, l.cfg.grid.y, l.cfg.grid.z });
            try std.testing.expectEqual(dims(v.get("block").?), [3]u32{ l.cfg.block.x, l.cfg.block.y, l.cfg.block.z });
            try std.testing.expectEqual(v.get("smem").?.integer, l.cfg.shared);
            try std.testing.expectEqual(v.get("pdl").?.integer != 0, l.cfg.pdl);
            var buf: [2048]u8 = undefined;
            try std.testing.expectEqualStrings(v.get("params").?.string, try masked(&l.args, &buf));
        }
        try std.testing.expect(lines.next() == null);
    }
}

test "decode passes each buffer where experts.cu does" {
    // the recording masks the addresses: which buffer goes where, by the kernels' parameter lists
    const c = cases[0];
    const ex = experts(c);
    const sc = scratch(c.slots);
    const ls = try xe.decodeLaunches(ex, sc, x, 5120, pick, wts, out, c.r, limit);
    const want = [3][]const u64{
        // decode_prep: x, pick, suh0, suh1, out0, out1, uids, ucount, members, wts, y, add (none), out, epoch
        &.{ x, pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu, sc.ids, sc.count, sc.members, wts, sc.y, 0, out, sc.epoch },
        // grouped_cp_kernel: X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z
        &.{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z },
        &.{ sc.xd, sc.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z },
    };
    for (&ls, want) |*l, ptrs| {
        var got: [16]u64 = undefined;
        var n: usize = 0;
        for (0..l.args.count) |i| if (l.args.sizes[i] == 8) {
            got[n] = l.args.integer(i).?;
            n += 1;
        };
        try std.testing.expectEqualSlices(u64, ptrs, got[0..n]);
    }
    const epi = struct {
        fn of(a: *const cuda.Args) xe.DecodeEpi {
            return std.mem.bytesToValue(xe.DecodeEpi, a.storage[a.offsets[a.count - 1]..][0..@sizeOf(xe.DecodeEpi)]);
        }
    };
    try std.testing.expectEqual(xe.DecodeEpi{ .pick = pick, .E = 385, .svh_g = ex.svh_g, .svh_u = ex.svh_u, .suh_d = ex.suh_d, .xd = sc.xd, .limit = limit, .act_mode = 1, .cnt = sc.cnt_gu, .ready = sc.ready, .ready_cnt = sc.ready_cnt, .epoch = sc.epoch, .discard = 1 }, epi.of(&ls[1].args));
    try std.testing.expectEqual(xe.DecodeEpi{ .pick = pick, .E = 385, .svh_d = ex.svh_d, .y = sc.y, .wts = wts, .out = out, .cnt = sc.cnt_d, .ready = sc.ready, .ready_cnt = sc.ready_cnt, .epoch = sc.epoch, .discard = 1 }, epi.of(&ls[2].args));
}
