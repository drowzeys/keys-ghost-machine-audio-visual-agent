"""keys Ghost Machine Studio: a film as a walk through a tree of clips.

The Ghost (the voice agent's face) stands on one node of the tree. At each step the Director writes two or three
distinct directions for what happens next, each rendered first as a cheap draft (a keyframe, then a short low-res
clip that opens on the last frame of the clip before). The person picks one by voice or by click; only the picked
branch is rendered at full quality, opening on its parent's last frame, and the Ghost moves onto it. Roads not taken
stay in the tree, dimmed, and any earlier node can be walked back to and branched from.

    tree.py      the tree model and its per-project store on disk
    writer.py    the Director's option shots (the brain's director lane), with a plain fallback
    service.py   the one-at-a-time render queue (renderer.room = the memory governor's admission), QC, export
    voice.py     what the person says to the Ghost ("take the left path", "mix A with C", "export this path")
    routes.py    the HTTP API, the page, and the hook the app server calls
    stub.py      a stand-in renderer (Pillow scenes + ffmpeg) for development without video models
"""
