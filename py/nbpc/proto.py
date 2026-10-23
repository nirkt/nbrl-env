"""ctypes mirror of work/loader/nbenv_proto.h (the nbhost <-> Python shared-memory protocol). Keep in sync."""
import ctypes as C

NBE_MAGIC = 0x3145424E
NBE_VERSION = 5
MAXP = 16
STATE_CAP = 64 * 1024
MAP_CAP = 16384
REC_CAP = 4096
# expertbot round: the frame slot size must match the game host binary: 512*256*4 for the training round's nbhost,
# 2400*1080*4 for work/loader/nbhost (full-resolution capture): set NB_PIX_CAP together with NBHOST_BIN
import os as _os
PIX_CAP = int(_os.environ.get("NB_PIX_CAP", str(2400 * 1080 * 4)))   # = NBE_PIX_CAP of loader/nbenv_proto.h (full-resolution frame slot)
OBS_VEC = 436
OBS_GRID = 6 * 25 * 25
CMD_NONE, CMD_STEP, CMD_RESET, CMD_QUIT, CMD_PING, CMD_SETUP, CMD_JOYTEST = 0, 1, 2, 3, 4, 5, 6
ST_BOOT, ST_READY, ST_RUNNING, ST_DEAD = 0, 1, 2, 9
F_STATE, F_PIXELS, F_REC, F_MAP, F_OBS = 1, 2, 4, 8, 16
OFF_CMD = 256
OFF_RESP = 4096
OFF_STATE = 64 * 1024
OFF_MAP = OFF_STATE + STATE_CAP * 4
OFF_REC = OFF_MAP + MAP_CAP * 4
OFF_PIX = OFF_REC + REC_CAP * 32
OFF_OBS = OFF_PIX + PIX_CAP
TOTAL = OFF_OBS + MAXP * (OBS_VEC + OBS_GRID) * 4


class Action(C.Structure):
    _fields_ = [("move_dir", C.c_int32), ("fire", C.c_int32), ("aim_deg", C.c_float), ("super_dist", C.c_float)]


class Div(C.Structure):
    """diversity round (2026-10-09): NbeCmd.div, the RESET's battle setup (global data ids; 0 = default)"""
    _fields_ = [("location", C.c_int32), ("theme", C.c_int32), ("chr", C.c_int32 * MAXP), ("skin", C.c_int32 * MAXP),
                ("gadget", C.c_int32 * MAXP), ("star", C.c_int32 * MAXP),   # cards.csv ids (gadget MetaType 5 / star power 4)
                ("setup_super_mask", C.c_uint32),                            # SETUP: bit i = fill player i's super
                ("setup_unprotect_mask", C.c_uint32),                        # SETUP: bit i = clear player i's spawn protection
                ("reserved", C.c_int32 * 2)]


DIV_MAGIC = 0x31564944   # Hdr.reserved[0] of a game host that understands Cmd.div / Resp.div_*


class Cmd(C.Structure):
    _fields_ = [("type", C.c_uint32), ("ticks", C.c_uint32), ("flags", C.c_uint32), ("seed", C.c_int32),
                ("ctl_mask", C.c_uint32), ("hold_mask", C.c_uint32), ("native_mask", C.c_uint32), ("obs_mask", C.c_uint32),
                ("act_tick", C.c_int32), ("gas", C.c_int32), ("pix_w", C.c_int32), ("pix_h", C.c_int32),
                ("reserved", C.c_int32 * 4), ("act", Action * MAXP),
                ("set_mask", C.c_uint32), ("ai_off_mask", C.c_uint32), ("set_x", C.c_int32 * MAXP), ("set_y", C.c_int32 * MAXP),
                # pctrain round (NBE_VERSION 5): "native + lead" aim for the native AI of players in lead_mask (work/loader/env.c lead_aim)
                ("lead_mask", C.c_uint32), ("lead_dt", C.c_float), ("lead_speed", C.c_float), ("lead_frac", C.c_float),
                # diversity round (appended): RESET battle setup (location / theme / per-player character and skin)
                ("div", Div)]


class Player(C.Structure):
    _fields_ = [(n, C.c_int32) for n in (
        "present", "char_id", "alive", "hp", "max_hp", "x", "y",
        "damage_dealt", "damage_taken", "gas_damage", "kills", "hits", "box_damage", "cubes", "cubes_total", "gas_ticks", "in_gas",
        "bomb_damage", "eliminated", "elim_rank", "elim_tick", "elim_gas", "leaping", "super_charge", "super_max", "ammo")] + [
        ("reserved", C.c_int32 * 4)]


class Resp(C.Structure):
    _fields_ = [("seq", C.c_uint32)] + [(n, C.c_int32) for n in (
        "status", "tick", "done", "result", "local_index", "generation", "alive", "hud_left", "local_rank", "winner")] + [
        ("map_hash", C.c_uint32)] + [(n, C.c_int32) for n in (
        "map_w", "map_h", "map_written", "state_ints", "rec_n", "rec_lost", "pix_bytes", "pix_w", "pix_h", "obs_written",
        "cam_valid")] + [("cam_x", C.c_float), ("cam_y", C.c_float)] + [(n, C.c_int32) for n in (
        "seed_applied", "seed_plus1", "steps_in_frame", "frame", "us_step", "us_state", "us_wait", "njumps")] + [
        ("jumps", (C.c_int32 * 5) * MAXP), ("ncubes", C.c_int32), ("cubes", (C.c_int32 * 2) * MAXP),
        ("reserved", C.c_int32 * 8), ("err", C.c_char * 128), ("pl", Player * MAXP),
        # expertbot round (work/loader/nbhost only; other binaries leave it 0): the world camera view-projection matrix
        # of the frame drawn since the previous response (column-major, render coords (x, -y, z))
        ("cam_mat", C.c_float * 16), ("cam_mat_seq", C.c_int32), ("cam_mat_n", C.c_int32),
        # diversity round (appended; written on RESET responses, read back from the game's objects): sim location, its
        # theme, the client's location (client mode), mode variation, rejected requests (bit i player i, 30 location,
        # 31 theme), per player the active loadout's character / skin data ids
        ("div_valid", C.c_int32), ("div_location", C.c_int32), ("div_theme", C.c_int32), ("div_cl_location", C.c_int32),
        ("div_variation", C.c_int32), ("div_err", C.c_int32), ("pl_char", C.c_int32 * MAXP), ("pl_skin", C.c_int32 * MAXP),
        ("div_reserved", C.c_int32 * 4)]


class Hdr(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in (
        "magic", "version", "total_size", "hdr_size", "off_cmd", "off_resp", "off_state", "off_map", "off_rec", "off_pix",
        "off_obs", "state_cap", "map_cap", "rec_cap", "pix_cap", "obs_vec", "obs_grid", "maxp",
        "cmd_seq", "resp_seq", "status", "heartbeat")] + [("pid", C.c_int32), ("boot_ms", C.c_int32), ("crashed", C.c_int32),
                                                         ("reserved", C.c_uint32 * 13)]


assert C.sizeof(Cmd) <= OFF_RESP - OFF_CMD and C.sizeof(Resp) <= OFF_STATE - OFF_RESP
