# MicroDuck third-party provenance and notices

## Scope and modifications

CASCADE's `microduck_policy.py` adapts the observation layout, exact HOME and
raw-action/target contract from Pollen Robotics' official runtime/RL sources.
`microduck_actuator.py` adapts the numerical XL330 firmware/motor law and M6
friction budget/parameters from BAM to validated, solver-independent NumPy
input/output. CASCADE adds fail-closed input/hash/schema checks, CPU-only lazy
ONNX loading, explicit battery/load history and a non-overwriting admission
CLI. These files have been modified; they are not unmodified upstream files
and are not a physically validated BAM port to Isaac.

Upstream BAM source attribution, retained verbatim:

> Copyright 2025 Marc Duclusaud & Grégoire Passault

The Apache-2.0 text below applies to the adapted upstream code; CASCADE's
repository-level license does not remove those third-party obligations.
External source `LICENSE` files and model-card declarations are independently
hash-pinned in `manifest.json` and preserved by the admission CLI.

## Immutable sources

| Material | Source revision |
|---|---|
| Pollen Robotics runtime contract | `pollen-robotics/microduck@1fa84386f07884e27866411bc1ba166977bced95` |
| Optional gait target transform (`robotd-targets-v1`) | `pollen-robotics/microduck@9136aa4ee88e81edf2bcaf3527e90b65da25f1eb`, `robotd/src/control.rs` and `duck-control/src/model.rs` |
| Pollen Robotics RL / MJCF / meshes | `pollen-robotics/microduck_rl@8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec` |
| Rhoban BAM code and calibrated XL330 M6 parameters | `Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967` |
| Official ONNX weight repository | `pollen-robotics/microduck-policies@d5a8b55033e157f1af2ed6bd5c1e435b770a8ee0` (Hugging Face) |

## Licenses are separate

- **Code:** runtime, RL and BAM each declare Apache-2.0 in their own LICENSE.
- **3D model files:** upstream RL README states literally **“Creative Commons
  BY-SA-NC”**, with **no version specified**. Do not assume 4.0, infer Apache
  from the code, or claim commercial/event permission. Conversion does not
  remove NC/SA obligations. Clarification belongs with upstream/rightsholders.
  Evidence: https://github.com/pollen-robotics/microduck_rl/blob/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec/README.md#L268-L271
- **Weights:** the weight repository's own model card independently declares
  `license: apache-2.0`. This is not inferred from code/model licenses.
  Evidence: https://huggingface.co/pollen-robotics/microduck-policies/blob/d5a8b55033e157f1af2ed6bd5c1e435b770a8ee0/README.md

No model, mesh, converted USD or weight bytes are committed here. The manifest
contains per-file byte counts and SHA-256s, immutable source revisions and
separate license identities. Fetch preserves the complete official model
subtree, including dependencies/ancillary scenes; it does not execute scripts
or promise that every scene/variant is supported by CASCADE.

## Explicit admission

From a source checkout, with a user-owned destination **outside** the checkout:

```sh
python scripts/microduck_assets.py --fetch --destination /absolute/external/microduck --accept-model-license
python scripts/microduck_assets.py --check --destination /absolute/external/microduck
```

`--source-directory /absolute/local/mirror` with `--fetch` uses only an existing
mirror in `<source id>/<upstream path>` layout. No network occurs on import,
`--check`, or local-mirror fetch. `--check` neither repairs nor creates files;
failed checks return nonzero. `--fetch` never overwrites an existing artifact;
a failed batch can retain earlier verified files, but never publishes a bad
partial download. It accepts no archives. Symlinked artifacts/ancestors are
rejected. The destination must not be concurrently edited by hostile processes.
`--accept-model-license` is an explicit acknowledgment, not a legal permission
grant. Admission and conversion/physics validation are separate operations.

Policy candidates are explicit: `velstand.onnx` with non-backlash
`robot_allcollisions.xml` is the planned initial pair. Listing all ten official
ONNX files does not enable rollers, sit/stand, kicks or other skills. Their
historical training recipe/geometry is not fully recoverable from metadata.
No automatic checkpoint replacement or model tuning is performed.

The separate `policy-candidates.json` pins the opt-in research checkpoint
`RemiFabre/microduck-rough-walk-e@fa7b27eeb5610d3b351362f4bd71691ee8be3d7d`.
Its [model card](https://huggingface.co/RemiFabre/microduck-rough-walk-e/blob/fa7b27eeb5610d3b351362f4bd71691ee8be3d7d/README.md)
declares Apache-2.0 for weights and explicitly reports simulation-only use.
No alternative weight bytes are committed. Selecting `--policy-profile
rough_walk_e` admits only that exact digest and size; it neither changes the
official default nor grants native locomotion or hardware admission. The
converted robot geometry remains governed by its separate upstream terms.

## Numerical boundary

`microduck_actuator.py` is a **numerical reference/golden validator only**, not
the preferred production integration. The planned Newton path should reuse
the native BAM actuator under audit in IsaacLab PR #8161, head
`28aa1fca5843208ff9a67935695a4d5376e44d50`, rather than duplicating `DriveBam`.
It requires the MJWarp solver and does not establish PhysX support.

The actuator returns commanded motor effort including back-EMF; separate
frictionloss budget, mechanical viscosity and armature must be implemented
with equivalent semantics by the solver. External load is the previous
`-bias + constraint - own DOF friction`, not this tick's freshly commanded
torque. Battery load uses previous commanded torque, separately from previous
solver-applied actuator torque. No static friction is approximated by a
`sign(dq)` function. Parameters are preserved, including the unused
identification `q_offset` (not a robot home correction).

`friction_reference` is also explicit: `bam_model` preserves the CPU
`Model.compute_frictions` quadratic sign/magnitude gates; `bam_mjlab`
preserves the training source's different quadratic convention (the PR
matches that convention). Do not label these as identical reference laws.

`max_current` is a required explicit constructor choice: BAM XL330 defaults to
1.75 A in the audited training constructor; `infer_policy.py` overrides it with
None. Those are different reference configurations, not established historical
training provenance. Retardation, backlash encoder sums, domain randomization,
constraint forces/contact handling and solver integration remain outside this
pure motor-law adapter. Standalone numeric/ONNX agreement proves none of
balance, locomotion, PhysX/Newton fidelity or sim-to-real transfer.

## Apache License 2.0 (unmodified text from BAM LICENSE)

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
