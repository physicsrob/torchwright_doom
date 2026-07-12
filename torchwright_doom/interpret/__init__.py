"""Output side: consume completed inference artifacts and frozen bundle data.

Never loads a model, never touches Transformers, makes no rendering
decisions: ``decode`` is last-write-wins blitting of the emitted protocol,
``compare`` scores against the pydoom oracle (``reference`` builds its
scene), ``artifacts`` writes the token dump, and ``formatter`` wraps the
portable prettifier kernel.
"""
