# PrismaQuant Weights License, Version 1.0

SPDX short identifier: LicenseRef-PrismaQuant-Weights-1.0

> **TL;DR.** Use these weights for anything, commercial included. The one thing
> you can't do is re-host copies of them. Exact copies on other sites split the
> download count, and the download count is how people judge a model's
> popularity. Download from Hugging Face, and have your applications pull from
> the original repository. If you don't like that condition, download
> PrismaQuant, make your own version, and release it wherever you like, crediting
> the tool.

This license grants broad rights to use, modify and build on the Weights,
commercially or otherwise. It adds three conditions: the Weights are published
from one place, the author is credited, and no one presents them as their own.
It is not an OSI-approved open-source license and should be described as an
open-weights license with conditions.

## 0. Terms supplied by the Licensor

Each release states these values in its model card:

    Author:               Robert Tand
    Contact:              robert.tand@icloud.com
    Original Repository:  https://huggingface.co/rdtand/<repository>
    Copyright:            Copyright (c) 2026 Robert Tand

## 1. Definitions

- **"Weights"** means the quantized model files, configuration and metadata
  published in the Original Repository, and their accompanying model card.
- **"Copy"** means the Weights, or any file set whose tensor contents are
  identical to them or recoverable from them by a lossless operation such as
  renaming, resharding, repacking, re-serialization or a change of container
  format. Identity of contents decides, however the files were obtained: a
  bit-exact reproduction of the Weights is a Copy.
- **"Derivative"** means a model materially transformed from the Weights, such
  as a fine-tune, a merge, a distillation, or a re-quantization to a different
  numeric format. A Copy is not a Derivative. A model you produce yourself from
  a base model, with PrismaQuant or any other tool, is not a Derivative of the
  Weights, and this license does not cover it unless its tensor contents are
  identical to the Weights, in which case it is a Copy.
- **"Distribution"** means any software, container image, installer, script,
  recipe, notebook, product or service that makes the Weights available to its
  users or runs them for others.
- **"Public Host"** means any service from which third parties can download
  files: model hubs (including other Hugging Face repositories), object
  storage, file-sharing and torrent services, CDNs, and container registries.

## 2. Grant

Subject to your compliance with Sections 3 to 6, the Licensor grants you a
worldwide, royalty-free, non-exclusive license to use, run, study, modify,
create Derivatives of, deploy, and offer products and services built on the
Weights, for any purpose, including commercial purposes.

## 3. One source: no mirrors

3.1 **No public re-hosting.** You must not upload, mirror or otherwise make a
Copy available on or through a Public Host. The Original Repository is the
only public source of the Weights.

3.2 **Distributions fetch from the Original Repository.** A Distribution may
deliver the Weights to its users only by downloading them from the Original
Repository at install or run time, and must link to the Original Repository in
its documentation.

3.3 **Private copies are fine.** You may keep Copies in local caches, internal
storage, private registries and air-gapped deployments, for use within your
organization, provided no third party can download them from you.

3.4 **Written permission.** The Licensor may permit a specific mirror in
writing. A permitted mirror must link to the Original Repository and carry
this license unchanged.

## 4. Attribution

4.1 Every Distribution, Derivative, deployment documentation, publication or
public presentation of results obtained with the Weights must state, in a
place users normally look (a README, model card, "About" or "Credits" screen):

> Quantized with PrismaQuant by Robert Tand —
> https://huggingface.co/rdtand/<repository>

4.2 You must retain this license, the copyright notice and the model card's
attribution in every Copy, and you must not remove or alter them.

4.3 A Derivative must name the Weights it was made from, with a link to the
Original Repository, in its model card metadata (`base_model`) and its README.

## 5. No misrepresentation

5.1 You must not present the Weights, a Copy, or a Derivative's quantization as
your own work or as the work of anyone other than the Author.

5.2 "PrismaQuant", "PrismaAURA", "PrismaSCOUT", "PrismaAQUA" and related names
identify the Author's work. You may use them only to give the attribution
required by Section 4 or to describe the Weights accurately. A Derivative must
not use them in its name, and nothing in this license grants any trademark
right.

## 6. Upstream terms

The Weights are derived from a base model released under its own license (for
example, MIT for GLM-5.3-Flash, Apache-2.0 for Qwen models). You must also
comply with that license, including retaining its notices. This license
restricts only the Licensor's own contribution; it does not limit any right the
upstream license grants in the base model itself.

## 7. Termination

7.1 Your rights under this license terminate automatically if you fail to
comply with it.

7.2 If you cure a first violation within 30 days of learning of it, your rights
are reinstated. Rights are not reinstated after a second violation.

7.3 A Copy made available in breach of Section 3 is unlicensed. The Licensor
may request its removal from any platform hosting it, including through that
platform's license-violation or takedown procedures.

## 8. Disclaimer

THE WEIGHTS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR OR
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE WEIGHTS OR THEIR USE.
