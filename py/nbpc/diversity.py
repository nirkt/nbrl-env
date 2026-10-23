"""Battle diversity (diversity round 2026-10-09): the controlled set of brawlers / skins / Solo Showdown maps / themes and
helpers for the RESET battle setup (NbeCmd.div; game host work/loader of the diversity round, Hdr.reserved[0] = 'DIV1').

Ids are global data ids = type * 1000000 + row (csv_logic, rows counted after the 2 header lines): characters 16,
skins 29, locations 15, location themes 47 (the theme's type number is the one the game uses for LocationData+0x58).
The tables are read from the extracted APK (instance.APK = apk_extract/ in this repo); names are resolved to ids at import.

    from nbpc import diversity as D
    div = D.make_div(location="Survival3", theme="MortuaryShowdown", chr=["Sniper", "Shaman", ...], skin=[0, 1, ...])
    inst.reset(seed, div=div)           # nbpc.instance; MAEnv: battle_fn returns cfg["div"] = div
    info = inst.battle_info()           # what the game built (read back from its objects)
    D.check_battle(div, info, inst.map_tiles())
    D.save_battle_map(path, info, inst.map_tiles())   # map.npy (int32 (3600,) like data/map_showdown.npy) + map.json
"""
import csv, json, os
import numpy as np

from .instance import APK as _APK
ASSETS = os.path.join(_APK, "assets")
T_CHAR, T_SKIN, T_LOC, T_THEME, T_TILE = 16, 29, 15, 47, 27


def _rows(name):
    with open(os.path.join(ASSETS, "csv_logic", name), newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    return [dict(zip(hdr, r)) for r in rows[2:]]


_cache = {}


def table(name):
    """{row name: (global id, row)} of a csv_logic table (named rows; the row index is the data index)"""
    if name not in _cache:
        T = {"characters.csv": T_CHAR, "skins.csv": T_SKIN, "locations.csv": T_LOC, "location_themes.csv": T_THEME, "tiles.csv": T_TILE}[name]
        out = {}; k = 0
        for r in _rows(name):
            n = r[next(iter(r))]
            if n:
                out[n] = (T * 1000000 + k, r); k += 1
        _cache[name] = out
    return _cache[name]


def name_of(gid):
    T = gid // 1000000
    f = {T_CHAR: "characters.csv", T_SKIN: "skins.csv", T_LOC: "locations.csv", T_THEME: "location_themes.csv", T_TILE: "tiles.csv"}.get(T)
    if not f: return None
    for n, (i, _) in table(f).items():
        if i == gid: return n
    return None


def cid(name): return table("characters.csv")[name][0]
def sid(name): return table("skins.csv")[name][0]
def lid(name): return table("locations.csv")[name][0]
def tid(name): return table("location_themes.csv")[name][0]


# ---------------------------------------------------------------- the chosen set (see work/APPLY.md for the reasons)
# brawler (characters.csv Name), display name, attack kind, alternate skin (skins.csv; a different model in the APK).
# Not in the set: Tick / Grom (BulletExplosion areas: no mines / secondary bullets in this client), Sprout (chain bullets),
# Barley (Dot puddle: works only with the game host's --area-dot, added after the set was chosen; can be added).
# Bull's super is a Charge: it needs --native-charge (the charge start 0xb475cc is a ret in this libg). BRAWLERS.md has
# every brawler.
ROSTER_INFO = [
    ("Sniper", "Piper", "straight bullet (long range, damage grows with travel)", "Sniperrose"),
    ("Gunslinger", "Colt", "straight bullets (burst of 6)", "GunSlingerRockstar"),
    ("ShotgunGirl", "Shelly", "pellets / shotgun (5 pellets)", "BanditGirlBandita"),
    ("BullDude", "Bull", "pellets / shotgun (5 pellets, short); super = charge (needs --native-charge, NOTES §1.4)", "BullGuyHair"),
    ("TntDude", "Dynamike", "lob (2 sticks) + area explosion", "TntGuySanta"),
    ("ArtilleryDude", "Penny", "splash shot; super summons a mortar that lobs shells", "ArtilleryGalBunny"),
    ("DeadMariachi", "Poco", "wide wave (piercing cone)", "PocoPirate"),
    ("RocketGirl", "Brock", "rocket (explodes, area)", "RocketGirlBeach"),
    ("Shaman", "Nita", "shock wave; super summons a bear", "ShamanPanda"),
    ("Mechanic", "Jessie", "bouncing orb; super summons a turret", "MechanicSummer"),
]
ROSTER = [r[0] for r in ROSTER_INFO]
ALT_SKIN = {r[0]: r[3] for r in ROSTER_INFO}
DEFAULT_BRAWLER = "Sniper"
# Solo Showdown locations (locations.csv: GameModeVariation Showdown, 60x60 maps, only standard tiles) and their own theme
LOCATIONS = ["Survival612", "Survival3", "Survival2", "Survival1"]
DEFAULT_LOCATION = "Survival612"
THEMES = ["ArcadeShowdown", "DefaultShowdown", "MortuaryShowdown", "MadEvilManorShowdown"]


def default_skin(brawler):
    return table("characters.csv")[brawler][1]["DefaultSkin"]


def make_div(location=None, theme=None, chr=None, skin=None):
    """-> NbeCmd.div dict of ids. location / theme: name or id (None = default); chr: per player index brawler name / id
    (None / 0 = default Sniper); skin per player index: skin name / id, or 0 / 1 = the brawler's default / alternate skin
    (ALT_SKIN), or (brawler, 0/1)."""
    def _id(x, f):
        if x is None or x == 0: return 0
        return int(x) if isinstance(x, (int, np.integer)) else f(x)
    out = {"location": _id(location, lid), "theme": _id(theme, tid), "chr": [0] * 16, "skin": [0] * 16}
    names = []
    for i, c in enumerate(chr or []):
        out["chr"][i] = _id(c, cid); names.append(c if isinstance(c, str) else (name_of(int(c)) if c else DEFAULT_BRAWLER))
    for i, s in enumerate(skin or []):
        if isinstance(s, tuple): b, s = s
        else: b = names[i] if i < len(names) else DEFAULT_BRAWLER
        if isinstance(s, str): out["skin"][i] = sid(s)
        elif s in (0, 1, False, True): out["skin"][i] = sid(ALT_SKIN[b]) if (s and b in ALT_SKIN) else 0
        else: out["skin"][i] = int(s)
    return out


def random_div(rng, roster=None, locations=None, themes=None, p_alt_skin=0.5, p_own_theme=0.5, players=10):
    """a random setup from the set: every player a random brawler, alternate skin with p_alt_skin, a random location, its own
    theme with p_own_theme else a random theme of the set"""
    roster = roster or ROSTER; locations = locations or LOCATIONS; themes = themes or THEMES
    br = [roster[int(rng.integers(len(roster)))] for _ in range(players)]
    sk = [int(rng.random() < p_alt_skin) for _ in range(players)]
    loc = locations[int(rng.integers(len(locations)))]
    th = None if rng.random() < p_own_theme else themes[int(rng.integers(len(themes)))]
    return make_div(location=loc, theme=th, chr=br, skin=list(zip(br, sk)))


def expected(div):
    """what a battle with this request must look like: location id, theme id, per player (character id, skin id)"""
    div = div or {}
    loc = div.get("location") or lid(DEFAULT_LOCATION)
    lname = name_of(loc)
    th = div.get("theme") or tid(table("locations.csv")[lname][1]["LocationTheme"])
    chars, skins = [], []
    for i in range(10):
        c = (div.get("chr") or [0] * 16)[i] or cid(DEFAULT_BRAWLER)
        s = (div.get("skin") or [0] * 16)[i] or sid(default_skin(name_of(c)))
        chars.append(c); skins.append(s)
    return {"location": loc, "theme": th, "chr": chars, "skin": skins}


def map_reference(location):
    """the location's map from maps.csv as tile data ids (H*W,) with spawn / box digits as Open (27000000), + the raw rows"""
    lname = name_of(location) if isinstance(location, (int, np.integer)) else location
    mname = table("locations.csv")[lname][1]["Map"]
    rows, cur = [], None
    for r in _rows("maps.csv"):
        if r["Map"]: cur = r["Map"]
        if cur == mname: rows.append(r["Data"])
        elif rows: break
    code = {}
    for n, (i, r) in table("tiles.csv").items():
        code.setdefault(r["TileCode"], i)
    ids = [code["."] if ch.isdigit() else code.get(ch, -1) for row in rows for ch in row]
    return np.array(ids, np.int32), rows


def check_battle(div, info, tiles):
    """request vs the game's read-back (Instance.battle_info) and the published tiles vs maps.csv -> dict of booleans"""
    if info is None:
        return {"div_supported": False}
    e = expected(div)
    ref, _ = map_reference(info["location"])
    t = np.asarray(tiles, np.int32)
    return {"location": info["location"] == e["location"], "theme": info["theme"] == e["theme"],
            "chr": info["chr"][:10] == e["chr"], "skin": info["skin"][:10] == e["skin"],
            "client_location": info["cl_location"] in (0, info["location"]),
            "err": info["err"] == 0, "map_vs_csv": bool(len(t) == len(ref) and (t == ref).all())}


def save_battle_map(folder, info, tiles, extra=None):
    """per-battle map publication: folder/map.npy (int32 (H*W,) tile data ids, row-major y*W+x, the format of
    data/map_showdown.npy) and folder/map.json (location / theme ids + names, map name, W, H, hash, per-player brawler /
    skin ids + names)"""
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "map.npy"), np.asarray(tiles, np.int32))
    lname = name_of(info["location"])
    j = {"location": info["location"], "location_name": lname, "map": table("locations.csv")[lname][1]["Map"] if lname else None,
         "theme": info["theme"], "theme_name": name_of(info["theme"]), "map_w": info["map_w"], "map_h": info["map_h"],
         "map_hash": info["map_hash"], "chr": info["chr"][:10], "chr_names": [name_of(c) for c in info["chr"][:10]],
         "skin": info["skin"][:10], "skin_names": [name_of(s) for s in info["skin"][:10]], "err": info["err"]}
    if extra: j.update(extra)
    json.dump(j, open(os.path.join(folder, "map.json"), "w"), indent=1)
    return j
