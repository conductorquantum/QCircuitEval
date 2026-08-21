Error Taxonomy
==============

QCircuitEval records a versioned, seven-axis error profile beside each
benchmark result. The profile is a set of evidence predicates, not a forced
single label. A failed program can violate several predicates at once.

The labels describe observed contract violations. They do not claim to recover
the model's hidden cause or intent. Every assignment must follow a persisted
benchmark status, semantic reason code, diagnostic, or parameter-case status.


Axes
----

The two execution axes are:

``generation_execution_reliability``
    The provider produced no candidate, the source did not compile, candidate
    execution failed, or exhaustive candidate replay raised an exception.
    Verifier exceptions and unsupported grader capabilities do not enter this
    axis.

``interface_observation_validity``
    The candidate violated the callable interface or the declared observation
    boundary. Examples include a wrong entry-point signature, malformed
    returned semantic object, wrong measurement register, or invalid output
    shape.

The three algorithmic axes are:

``construction_resource_fidelity``
    The candidate violated a declared construction recipe or resource
    requirement. Examples include wrong qubit count, gate family, operation
    count, layer count, parameter binding, or optimization budget.

``interaction_lifecycle_fidelity``
    The candidate violated a declared relation between subsystems or a required
    lifecycle step. Examples include missing interaction edges, controlled
    corrections, argument-conditioned gates, entangling uncomputation, or
    connected interaction groups.

``shortcut_provenance_violation``
    The candidate used a prohibited route that can inject or compute the answer
    without the requested construction. Examples include returned
    probabilities, direct state preparation, dense unitary gates, prohibited
    solvers, or forbidden source calls.

The two semantic axes are:

``behavioral_target_mismatch``
    A decisive verifier comparison exceeded the contracted bound or produced a
    counterexample. This includes state, distribution, operator, channel,
    instrument, classical-I/O, objective, and certified approximation checks.

``parameter_domain_robustness``
    At least one recorded parameter case passed and at least one recorded case
    failed semantically. All-failing domains do not satisfy this predicate.
    Case-status counts are stored before reconciliation, so this label does
    not infer variation from an aggregate failure code.


Outcomes and exclusions
-----------------------

``verified_pass`` records have no error axes. ``observed_error`` records have one
or more classified axes, or an unclassified decisive failure reason.
``grader_nondecision`` records describe verifier inability or invalid grader
configuration. Benchmark ``infrastructure_error`` status also maps to
``grader_nondecision`` (with reason ``benchmark_status:infrastructure_error``)
rather than a model-error axis. ``resource_limit`` is separate because exceeding
a verification budget does not prove candidate error. ``ungraded`` covers
records without usable semantic evidence.

Accepted convention variants are not error axes. They belong in a separate
grader-disagreement analysis. Grader exceptions, unsupported lowering, route
errors, uncertainty bands, and contract defects are also excluded from model
error spokes. ``observed_error`` is neutral about whether the model, provider,
or execution environment caused a generation failure.

Unknown decisive reason codes remain in ``unclassified_reason_codes``. The
classifier never guesses a category from free-form text. This makes mapping
coverage measurable and prevents taxonomy changes from rewriting raw evidence.


Radar statistic
---------------

For axis :math:`a`, let :math:`A_i` be the set assigned to record :math:`i`.
For a plotted stratum containing :math:`N` assigned records, report

.. math::

   r_a = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[a \in A_i].

Every spoke uses the same denominator, ``all_assigned_records_in_stratum``.
Therefore :math:`0 \le r_a \le 1`. The axes overlap, so their counts can exceed
the number of failed records and their rates need not sum to one.

``classification_coverage`` uses a different, explicit denominator. It is the
fraction of ``observed_error`` records with at least one assigned axis. Reports
also retain unclassified reason counts and taxonomy-version counts. Paper plots
must stratify incompatible taxonomy versions instead of pooling them.

Repeated samples and feedback attempts are separate assigned records unless a
paper analysis declares another unit. Confidence intervals must use that
declared sampling unit. The raw summary does not impose an independence model.


Recorded shape
--------------

Each benchmark record contains ``error_taxonomy``:

.. code-block:: json

   {
     "taxonomy_version": "1",
     "multi_label": true,
     "outcome": "observed_error",
     "axes": [
       "construction_resource_fidelity",
       "behavioral_target_mismatch"
     ],
     "reason_codes": ["metric_exceeds_fail_bound"],
     "grader_reason_codes": [],
     "unclassified_reason_codes": [],
     "parameter_case_status_counts": {
       "verified_pass": 0,
       "semantic_fail": 0,
       "execution_error": 0,
       "resource_limit": 0
     }
   }

The raw reason codes remain the audit trail. The versioned axes are a derived
analysis layer. Resumed historical records preserve a recorded taxonomy. Older
records without the field derive version 1 from their stored evidence.
