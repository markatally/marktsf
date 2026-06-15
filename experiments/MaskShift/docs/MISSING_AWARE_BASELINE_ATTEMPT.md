# Missing-aware Baseline Audit

Initial M10 audit found no S4M or ChannelTokenFormer-compatible implementation in the local `external/` tree. M11 then added the official ChannelTokenFormer repository at `external/ChannelTokenFormer` and ran `ChannelTokenFormer_missing` under the MaskShift encoder-mask protocol. M12 added the official S4M repository at `external/S4M` and ran the S4M model class under the same MaskShift encoder-mask protocol over three seed offsets. M14 then reran S4M in a larger reduced setting with 16 channels and doubled train/test windows.

Current status:

- official modern TSF architecture adaptation: PatchTST and TimeXer model classes imported from pinned TSLib revision `4e938a1`;
- official missing-aware architecture adaptation: ChannelTokenFormer_missing imported from official ChannelTokenFormer revision `b1c100e`;
- official missing-aware contrastive adaptation: S4M imported from official S4M revision `a718823` and evaluated over three seed offsets in M12, plus a larger reduced 16-channel M14 scale validation; local device-port patch changes one hard-coded `.cuda()` memory fetch to `.to(Q.device)` for MPS/CPU compatibility;
- lite missing-aware proxy: GRU-DLite in M6;
- remaining gap: none of these runs is a full paper-protocol reproduction; all are MaskShift-protocol adaptations with reduced local compute.

Do not state that MaskShift has exhaustive missing-specific baseline coverage. It has two official missing-aware architecture adaptations (CTF_missing and S4M), not the full CTF practical/irregular benchmark protocol and not the full S4M benchmark protocol.
