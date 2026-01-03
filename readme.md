hybrid_centrifugal_compressor/
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── compressor.py          # Main compressor class (RadComp style)
│   │   ├── solver.py              # Nonlinear solver (TurboFlow style)
│   │   └── problem.py             # Problem formulation
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── inducer.py             # From RadComp
│   │   ├── impeller.py            # From RadComp with TurboFlow losses
│   │   ├── diffuser.py            # Combined approach
│   │   └── volute.py              # Optional volute
│   │
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── geometry.py            # RadComp geometry structure
│   │   ├── validator.py           # Geometry validation
│   │   └── meridional.py          # Meridional plane calculations
│   │
│   ├── conditions/
│   │   ├── __init__.py
│   │   ├── operating. py           # Operating conditions (RadComp)
│   │   └── thermodynamics.py      # CoolProp wrapper
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── registry.py            # TurboFlow loss registry
│   │   ├── config.py              # Loss model configuration
│   │   ├── context.py             # Loss calculation context
│   │   │
│   │   ├── impeller/
│   │   │   ├── __init__.py
│   │   │   ├── incidence.py       # Multiple correlations
│   │   │   ├── skin_friction.py   # Jansen, Coppage, etc.
│   │   │   ├── blade_loading.py   # Rodgers, Aungier, etc.
│   │   │   ├── clearance.py       # Jansen, Brasz
│   │   │   ├── disc_friction.py   # Daily & Nece
│   │   │   └── recirculation.py   # Various models
│   │   │
│   │   └── diffuser/
│   │       ├── __init__.py
│   │       ├── friction.py
│   │       └── mixing.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── performance.py         # Performance analysis
│   │   ├── map_generation.py      # Performance map
│   │   └── optimization. py        # Design optimization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py
│       └── plotting.py
│
├── examples/
├── tests/
├── docs/
├── setup.py
└── requirements.txt