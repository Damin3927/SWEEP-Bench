# Third-party notices

The code in this repository is released under the MIT license ([LICENSE](LICENSE)), except the files listed below.
They are adapted from upstream projects, or are patches to upstream files, and they keep the license of the project
they come from. All of them were modified for this benchmark.

The git submodules under `policy_servers/*/upstream` are not part of this repository. They are fetched from their
upstream repositories at the pinned commits and keep their own licenses.

## Policy servers: upstream code and pretrained weights

- `policy_servers/cosmospolicy`: code Apache-2.0; base weights `nvidia/Cosmos-Predict2-2B-Video2World` under the
  NVIDIA Open Model License (gated on Hugging Face); the VAE from `Wan-AI/Wan2.1-T2V-1.3B` and the T5-11B text
  encoder, both Apache-2.0. Upstream's task checkpoints (NVIDIA non-commercial license) are not used.
- `policy_servers/dynamicvla`: upstream's code and the `hzxie/dynamic-vla-DOM` weights are under the S-Lab License
  1.0, which allows non-commercial use only; its language model, `HuggingFaceTB/SmolLM2-360M`, is Apache-2.0.
- `policy_servers/fastwam`: code MIT; the Wan2.2-TI2V-5B components (`Wan-AI/Wan2.2-TI2V-5B`) Apache-2.0.
- `policy_servers/groot`: code Apache-2.0; the weights' Hugging Face repository ships an NVIDIA license limited to
  non-commercial research and evaluation (its model card names the NVIDIA Open Model License). The backbone
  `nvidia/Cosmos-Reason2-2B` is gated: accept its terms on Hugging Face before the first download.
- `policy_servers/openpi`: code Apache-2.0; the base checkpoints (`gs://openpi-assets`) build on PaliGemma, so the
  Gemma Terms of Use apply to them as well (upstream's `LICENSE_GEMMA.txt`).
- `policy_servers/openvlaoft`: code MIT; `openvla/openvla-7b` is MIT on its model card and, as a Llama 2 derivative,
  also under the Llama 2 Community License (upstream's OpenVLA README).
- `policy_servers/vlajepa`: code Apache-2.0; the pretrained `ginwind/VLA-JEPA` checkpoint and
  `Qwen/Qwen3-VL-2B-Instruct` Apache-2.0; `facebook/vjepa2-vitl-fpc64-256` MIT.

## DynamicVLA: S-Lab License 1.0 (non-commercial)

From [hzxie/DynamicVLA](https://github.com/hzxie/DynamicVLA) at commit a27a06d. Copyright 2026 S-Lab.

- `policy_servers/dynamicvla/cuda_graph.py`: `GraphPolicy._sample_actions`, `_denoise_step`, `_embed_prefix` and
  `_embed_suffix`, adapted from `policies/dynamicvla/modeling_dynamicvla.py`. They also contain LeRobot code (see
  LeRobot below).
- `policy_servers/dynamicvla/finetune/patches/*.patch`: changes to `core/train.py`, `run.py` and
  `utils/instruction_generator.py`.
- `policy_servers/dynamicvla/finetune/*.yaml`: training configurations based on `configs/dynamicvla.yaml`.

The S-Lab License allows redistribution and use for non-commercial purposes only. For commercial use, contact the
DynamicVLA authors.

```text
S-Lab License 1.0

Copyright 2026 S-Lab

Redistribution and use for non-commercial purpose in source and
binary forms, with or without modification, are permitted provided
that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

In the event that redistribution and/or use for commercial purpose in
source or binary forms, with or without modification is required,
please contact the contributor(s) of the work.
```

## openpi: Apache License 2.0

From [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) at commit 650c5b0, by Physical
Intelligence. openpi ships no NOTICE file at that commit. The license text is at the end of this file.

- `policy_servers/openpi/finetune/grad_accum.py`: the train step that `make_train_step` builds, adapted from
  `train_step` in `scripts/train.py`, with gradient accumulation added.
- `policy_servers/openpi/serve.py`: `_FramePushServer._handler`, adapted from `WebsocketPolicyServer._handler` in
  `src/openpi/serving/websocket_policy_server.py`.
- `policy_servers/{cosmospolicy,dynamicvla,fastwam,groot,openvlaoft,vlajepa}/serve.py`: `_pack_array` and
  `_unpack_array`, adapted from `packages/openpi-client/src/openpi_client/msgpack_numpy.py`, and the websocket server
  loop and its error reply, which follow `WebsocketPolicyServer` in `src/openpi/serving/websocket_policy_server.py`.

## LeRobot: Apache License 2.0

From [huggingface/lerobot](https://github.com/huggingface/lerobot) v0.3.3, the version DynamicVLA requires.
Copyright 2025 HuggingFace Inc. team; Copyright 2025 Physical Intelligence and The HuggingFace Inc. team; Copyright
2024 The HuggingFace Inc. team. LeRobot ships no NOTICE file. The license text is at the end of this file.

- `policy_servers/dynamicvla/cuda_graph.py`: the four functions listed under DynamicVLA, whose DynamicVLA originals
  are based on LeRobot's SmolVLA and pi0 policies (`src/lerobot/policies/smolvla/modeling_smolvla.py` and
  `src/lerobot/policies/pi0/modeling_pi0.py`).
- `policy_servers/fastwam/patches/0001-observation-history.patch` and `0002-conveyor-datasets.patch`: the changes to
  Fast-WAM's copies of LeRobot files (`src/fastwam/datasets/lerobot/lerobot/datasets/video_utils.py` and
  `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py`), which carry LeRobot's Apache 2.0 header.

## Fast-WAM: MIT

From [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) at commit 45d8e14.

- `policy_servers/fastwam/patches/*.patch`: changes to Fast-WAM files. The changes to its copies of LeRobot files are
  under Apache 2.0 (see LeRobot above).

```text
MIT License

Copyright (c) 2026 The FastWAM Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Except for third-party code and files that retain their original license
headers, this repository is released under the MIT License.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenVLA-OFT: MIT

From [moojink/openvla-oft](https://github.com/moojink/openvla-oft) at commit e4287e9.

- `policy_servers/openvlaoft/finetune/patches/*.patch`: changes to `prismatic/vla/datasets/datasets.py` and
  `prismatic/vla/constants.py`.

```text
MIT License

Copyright (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## VLA-JEPA: Apache License 2.0, and starVLA: MIT

From [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA) at commit ec90777. Copyright 2026 VLA-JEPA Authors. The
Apache 2.0 text is at the end of this file.

- `policy_servers/vlajepa/finetune/bench_vlajepa.yaml`: training configuration based on
  `examples/SimplerEnv/train_files/vlajepa_ft.yaml` and `scripts/configs/vlajepa_libero_ft.yaml`.
- `policy_servers/vlajepa/finetune/patches/0001-qwen3-honor-configured-attn-implementation.patch`: changes to
  `starVLA/model/modules/vlm/QWen3.py`, which carries the starVLA header "Copyright 2025 starVLA community" and the MIT
  license. VLA-JEPA's `THIRD_PARTY_NOTICES.md` keeps such starVLA portions under MIT and its own changes under
  Apache 2.0. The MIT notice as VLA-JEPA ships it:

```text
MIT License

Copyright (c) StarVLA Team.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apache License 2.0

The license of openpi, LeRobot and VLA-JEPA, as shipped by openpi.

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
