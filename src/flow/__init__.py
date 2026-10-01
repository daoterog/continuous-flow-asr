"""The probability path: its coefficients, its time reparameterization, and sampling.

Each pure module is paired with the stateful driver that runs it against a model —
``schedule`` with ``tau_scheduler``, ``sampler`` with ``sampling_probe``. The drivers
know about the LightningModule; the modules they drive know only tensors.

Nothing is re-exported here on purpose: the drivers import the pure modules, so a
re-export would make this package's import order load-bearing.
"""
