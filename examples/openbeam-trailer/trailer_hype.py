"""The OpenBeam beta trailer, rebuilt on juice.motion (the original was ~1,500 lines of one-off code).

    juice motion trailer.py --preview                 480p check in seconds
    juice motion trailer.py --stills                  a frame from every scene
    juice motion trailer.py --orientation portrait    the same script, 9:16
"""
from pathlib import Path

from juice.motion import Video, scenes as s

HERE = Path(__file__).resolve().parent
SRC = Path.home() / "Downloads/openbeam-trailer/src"
REC = Path.home() / "Downloads/openbeam-trailer/recovered"
v = Video(length="short", brand=HERE / "juice.json")
ba, aa = SRC / "ba_RESEARCH.mov", SRC / "aa_RESEARCH.mov"

# Voice: "OpenBeam is now in beta." / "We're onboarding..." / the research line / the sign-up.
v.voice(ba, 0.00, 1.62, at=0.00)
v.voice(aa, 0.00, 2.36, at=1.95)
research = v.voice(ba, 1.74, 5.92, at=8.62)
v.voice(ba, 5.98, 7.62, at=13.98)
signup = v.voice(ba, 7.70, 11.42, at=19.12)

# 0.00 hook on camera
v.scene(s.Footage(v.source(ba, 0, 1.62), zoom=(1.0, 1.07)), 0.00, 1.62, energy=0)
v.overlay(s.Bug(), 0.0, 1.62)
v.overlay(s.Headline("Now in beta.", highlight={"beta.": "accent"}), 0.30, 1.62)
# 1.62 logo
v.scene(s.LogoReveal(badge="beta", tagline="Soft-body physics · in the cloud"), 1.62, 3.96, transition="flash", energy=1)
# 3.62 the app
v.scene(s.Panels([v.source(SRC / "rec_sep30.mov", 0.3, 2.5), v.source(REC / "rec_552313.mp4", 2.0, 4.2, crop=(790, 60, 2250, 1266))],
                 callouts=[(1, 0.22, 0.47, "Soft-body physics on a cloud GPU"), (1, 0.36, 0.11, "Streamed to a browser tab"),
                           (1, 0.80, 0.50, "4-hour sessions")], chip="NOW ONBOARDING · BETA"),
        3.62, 5.62, transition="logo", energy=2)
# 5.62 the flip: build, slow-mo, landing
flip = [(SRC / "rec_f2.mp4", 6.62, 7.22, 1.0, 5.62, (1.0, 1.06)), (SRC / "rec_cz.mp4", 0.30, 1.10, 1.0, 6.22, (1.04, 1.14)),
        (SRC / "rec_cz.mp4", 1.10, 1.60, 0.5, 7.02, (1.12, 1.22)), (SRC / "rec_cz.mp4", 1.60, 2.20, 1.0, 8.02, (1.08, 1.0))]
for path, a, b, speed, at, zoom in flip:
    v.scene(s.Footage(v.source(path, a, b, speed), zoom=zoom, cy=0.48), at, at + (b - a) / speed, energy=3)
v.overlay(s.Letterbox(), 6.22, 8.62)
v.overlay(s.Label("Soft-body physics simulator"), 5.82, 8.62)
v.overlay(s.Label("0.5×", size=26, color="accent", align="r"), 7.02, 8.02)
v.overlay(s.Slam(["SOFT-BODY", "PHYSICS."], pos=(0.08, 0.5), size=170, align="l"), 7.06, 8.02)
v.hit(8.42, "land")
# 8.62 research areas, each word slammed as it's said
world, robot, ai = research.out(3.33) - 0.03, research.out(4.24) - 0.03, research.out(5.05) + 0.05
v.scene(s.Footage(v.source(REC / "rec_173450.mp4", 60.0, 62.8, crop=(120, 146, 2544, 1426)), zoom=(1.0, 1.12), cy=0.55),
        8.62, 11.20, transition="flash", energy=3)
v.overlay(s.Tracker((0.37, 0.37, 0.23, 0.41), "Vehicle · soft-body"), 8.87, world)
v.overlay(s.Words([(w, research.out(t)) for w, t in (("If", 1.95), ("your", 2.12), ("research", 2.30), ("involves", 2.66))],
                  until=world), 8.62, world)
v.overlay(s.Slam(["WORLD", "MODELS"], size=230), world, 11.20)
v.scene(s.Footage(v.source(SRC / "rec_ob.mp4", 17.0, 18.1), zoom=(1.15, 1.04), vignette=0.5), 11.20, 12.20, transition="whip", energy=3)
v.overlay(s.Slam("ROBOTICS", pos=(0.06, 0.79 if v.orientation == "landscape" else 0.66), size=220, align="l"), robot, 12.20)
v.scene(s.Linked(v.source(REC / "rec_552313.mp4", 330, 331.9, crop=(236, 200, 1300, 1520)),
                 v.source(REC / "rec_552313.mp4", 330, 331.9, crop=(1560, 670, 1484, 834)),
                 label="MCP", left_label="AI agent", sub="Connect Codex or Claude to the sim over MCP"),
        12.20, 13.98, transition="whip", energy=3)
v.overlay(s.Slam("PHYSICAL AI", pos=(0.5, 0.14), size=150), ai, 13.98)
for t in (world, robot, ai):
    v.hit(t, "slam")
# 13.98 back on camera
v.scene(s.Footage(v.source(ba, 5.98, 7.62), zoom=(1.04, 1.10)), 13.98, 15.62, energy=1)
# 15.62 features
v.scene(s.FeatureGrid("A physics lab in a browser tab.", kicker="OpenBeam Cloud · beta", cards=[
    s.Card("Soft-body physics on a cloud GPU", v.source(SRC / "rec_sep30.mov", 100.0, 103.6)),
    s.Card("Bring your own maps and vehicles", v.source(SRC / "rec_sep30.mov", 9.0, 12.6, crop=(560, 140, 1820, 1160))),
    s.Card("Every physics step, recorded", v.source(SRC / "rec_ob.mp4", 21.0, 24.6), rec=True),
    s.Card("Connect Codex or Claude", body="Add this link as a remote MCP server in your AI app.",
           url="https://cloud.openbeam.world/mcp", button="Copy MCP link")]),
    15.62, 19.12, transition="flash", energy=3)
for k in range(4):
    v.hit(15.77 + k * 0.25, "click")
# 19.12 sign up
v.scene(s.Footage(v.source(ba, 7.70, 9.29), zoom=(1.0, 1.06)), 19.12, 20.71, energy=1)
url0, url1 = signup.out(8.45), signup.out(9.17)
v.overlay(s.UrlBar("openbeam.world", (url0, url1)), 19.12, 20.71)
v.hit(url0, "type", until=url1)
v.scene(s.Notification(v.source(ba, 9.29, 11.42), "Your account is ready.",
                       body="Soft-body physics on a cloud GPU, streamed to your browser.", button_label="Play OpenBeam",
                       sender="OpenBeam Cloud", button_at=signup.out(10.89) - 0.1 - 20.71), 20.71, 23.12, energy=1)
# 23.12 end card
v.scene(s.EndCard(line="Sign up at", badge="beta", sub="World models · Robotics · Physical AI"), 23.12, 27.12,
        transition="flash", energy=0)
v.music("hype", beat0=1.62)

