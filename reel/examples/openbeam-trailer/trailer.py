"""The OpenBeam beta trailer in the clean look: cuts on the beat, type that rises into place, the
product in flat windows, one accent color, a futuristic bed, the voice isolated. (trailer_hype.py is the loud cut.)

    juice motion trailer.py --preview                 480p check in seconds
    juice motion trailer.py --stills                  a frame from every scene
    juice motion trailer.py --orientation portrait    the same script, 9:16
"""
from pathlib import Path

from juice.motion import Video, clean as c

HERE = Path(__file__).resolve().parent
SRC = Path.home() / "Downloads/openbeam-trailer/src"
REC = Path.home() / "Downloads/openbeam-trailer/recovered"
v = Video(length="short", brand=HERE / "juice.json")
ba, aa = SRC / "ba_RESEARCH.mov", SRC / "aa_RESEARCH.mov"
v.music("future", beat0=1.62)
beat = v.beat   # 120 bpm from the logo: beat(k) = 1.62 + k/2

# Voice: "OpenBeam is now in beta." / "We're onboarding..." / the research line / the sign-up.
v.voice(ba, 0.00, 1.62, at=0.00)
v.voice(aa, 0.00, 2.36, at=1.95)
research = v.voice(ba, 1.74, 5.92, at=8.62)
v.voice(ba, 5.98, 7.62, at=13.98)
signup = v.voice(ba, 7.70, 11.42, at=19.12)

# 0.00 on camera
v.scene(c.Footage(v.source(ba, 0, 1.62), zoom=(1.0, 1.04), vignette=0.15), 0.00, beat(0), energy=0)
v.overlay(c.Title("Now in beta.", kicker="OpenBeam", highlight={"beta.": "accent"}, scrim=True), 0.30, beat(0))
# 1.62 the lockup, on the first downbeat
v.scene(c.Lockup(line="Soft-body physics, in the cloud."), beat(0), beat(4), energy=1)
v.hit(beat(0), "sub")
# 3.62 the product, two clips in one window
copy = dict(kicker="OpenBeam Cloud", title=["Soft-body physics", "on a cloud GPU."],
            body=["Streamed to a browser tab.", "Four-hour sessions."], url="cloud.openbeam.world")
v.scene(c.Frame(v.source(REC / "rec_552313.mp4", 2.0, 3.2, crop=(790, 60, 2250, 1266)), **copy),
        beat(4), beat(6), transition="wipe", energy=1)
v.scene(c.Frame(v.source(SRC / "rec_sep30.mov", 110.0, 111.2), enter=False, **copy), beat(6), beat(8), energy=1)
# 5.62 the flip: cut on every beat, the flip itself at half speed
v.hit(beat(8), "swell", until=beat(5))
v.hit(beat(8), "sub")
flip = [(SRC / "rec_f2.mp4", 6.30, 1.0, (1.0, 1.05)), (SRC / "rec_cz.mp4", 0.10, 1.0, (1.02, 1.08)),
        (SRC / "rec_cz.mp4", 0.80, 0.5, (1.06, 1.14)), (SRC / "rec_cz.mp4", 1.60, 1.0, (1.08, 1.02)),
        (SRC / "rec_f2.mp4", 9.20, 1.0, (1.0, 1.04))]
k = 8
for path, a, speed, zoom in flip:
    beats = 2 if speed < 1 else 1
    v.scene(c.Footage(v.source(path, a, a + beats * 0.5 * speed, speed), zoom=zoom, cy=0.48, vignette=0.15),
            beat(k), beat(k + beats), energy=3)
    k += beats
v.overlay(c.Caption("Soft-body physics, in real time", index="01"), beat(8) + 0.2, beat(14) - 0.05)
v.hit(beat(12), "sub", 0.6)
# 8.62 research areas: the footage cuts on each spoken term
world, robot, ai = research.out(3.33) - 0.03, research.out(4.24) - 0.03, research.out(5.05) + 0.05
drive = dict(crop=(120, 146, 2544, 1426))
v.scene(c.Footage(v.source(REC / "rec_173450.mp4", 40.0, 41.0, **drive), zoom=(1.0, 1.05), vignette=0.15),
        beat(14), beat(15.6), energy=2)
v.scene(c.Footage(v.source(REC / "rec_173450.mp4", 100.0, 101.0, **drive), zoom=(1.04, 1.0), vignette=0.15),
        beat(15.6), world, energy=2)
v.scene(c.Footage(v.source(REC / "rec_173450.mp4", 60.0, 61.0, **drive), zoom=(1.0, 1.06), cy=0.55, vignette=0.15),
        world, robot, energy=2)
v.scene(c.Footage(v.source(SRC / "rec_ob.mp4", 22.0, 22.6), zoom=(1.05, 1.0), vignette=0.15), robot, robot + 0.45, energy=2)
v.scene(c.Footage(v.source(SRC / "rec_ob.mp4", 34.4, 35.0), zoom=(1.0, 1.05), vignette=0.15), robot + 0.45, ai, energy=2)
v.scene(c.Split(v.source(REC / "rec_552313.mp4", 330, 332.2, crop=(236, 200, 1300, 1520)),
                v.source(REC / "rec_552313.mp4", 330, 332.2, crop=(1560, 670, 1484, 834)),
                link="MCP", left_label="AI agent", right_label="OpenBeam"), ai, 13.98, energy=2)
v.overlay(c.Terms([(w, research.out(t)) for w, t in (("If", 1.95), ("your", 2.12), ("research", 2.30), ("involves", 2.66))],
                  [("World models", world), ("Robotics", robot), ("Physical AI", ai)]), beat(14), 13.98)
for t in (world, robot, ai):
    v.hit(t, "tick")
# 13.98 back on camera
v.scene(c.Footage(v.source(ba, 5.98, 7.62), zoom=(1.04, 1.08), vignette=0.15), 13.98, beat(28), energy=1)
# 15.62 what's in it, a cell on every beat
v.scene(c.Bento("A physics lab in a browser tab.", kicker="OpenBeam Cloud · Beta", every=0.5, cards=[
    c.Card("Soft-body physics on a cloud GPU", v.source(SRC / "rec_sep30.mov", 100.0, 103.6)),
    c.Card("Bring your own maps and vehicles", v.source(SRC / "rec_sep30.mov", 9.0, 12.6, crop=(560, 140, 1820, 1160))),
    c.Card("Every physics step, recorded", v.source(SRC / "rec_ob.mp4", 21.0, 24.6), rec=True),
    c.Card("Connect Codex or Claude", body="Add this link as a remote MCP server in your AI app.",
           url="https://cloud.openbeam.world/mcp", button="Copy MCP link")]),
    beat(28), 19.12, energy=3)
for i in range(4):
    v.hit(beat(28) + 0.15 + i * 0.5, "tick")
# 19.12 sign up: the link as it's said, then a punch-in as the account comes through
v.scene(c.Footage(v.source(ba, 7.70, 9.29), zoom=(1.0, 1.04), vignette=0.15), 19.12, 20.71, energy=1)
v.overlay(c.Title("openbeam.world", kicker="Sign up at", highlight={"openbeam.world": "accent"}, scrim=True),
          signup.out(8.45) - 0.1, 20.71)
v.scene(c.Footage(v.source(ba, 9.29, 11.42), zoom=(1.2, 1.24), cy=0.38, vignette=0.15), 20.71, 23.12, energy=1)
v.overlay(c.Notice("Your account is ready.", body="Soft-body physics on a cloud GPU, in your browser.",
                   sender="OpenBeam Cloud"), signup.out(10.55), 23.12)
v.hit(signup.out(10.55), "tick")
# 23.12 end
v.scene(c.Lockup(line="Sign up at", url=True, sub="World models · Robotics · Physical AI"), 23.12, 27.12,
        transition="fade", energy=0)
v.hit(23.12, "sub", 0.7)
