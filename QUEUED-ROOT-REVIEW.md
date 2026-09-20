# Queued root review of this WIP (read, preserved; NOT yet addressed)

Source: /home/rob/tmp/cache-takeover-20260920/pq-render-root-review-queued.md
(review of checkpoint 3bc998d54d). The WIP is not accepted. Address each
item when root resumes this branch; nothing below is done yet.

1. **Double serialization before prewrite.** `torch.save(canonical,
   io.BytesIO())` serializes the whole output only to count its length
   BEFORE prewrite, then the writer serializes again. Replace with a
   conservative serialization bound from tensor/storage metadata
   (including actual serializer overhead), admitted before the one real
   write; reconcile actual bytes afterwards through existing contracts.
   Never pretend exact length is known without writing.
2. **Unaccounted temp footprint.** The atomic writer's temporary file +
   rename means `temp: 0` and the final path alone do not describe the
   actual footprint. Extend/reuse the EXISTING atomic writer to admit the
   actual temp/final lifetime and bounded peak; no second writer.
3. **Whole-file DEV hash tax.** Descriptor creation rereads the file via
   `read_bytes` and hashes it -- Rob forbids this DEV tax. The PB author
   is independently removing the mandatory output-digest prerequisite
   (existing null-digest path). Use that for DEV plus cheap current
   size/change evidence; no whole-payload reread/hash, no fake
   certification. Re-read the updated PB signatures when resuming.
4. **Weak cache-hit/retry semantics.** `Path.exists()` is not proof a
   render is reusable or that an earlier failed publication completed.
   Reuse the cache's existing validity/identity checks; a durable write
   followed by a publication failure must retry/reconcile publication,
   never answer a bogus cache-hit success. Keep stable
   batch/generation/descriptors across retry via existing state.
5. **~~SDK source preference~~ — WITHDRAWN by root (16:08).** The actual
   `staged_lease.py` diff is a candidate PIN/comment change only, NOT an
   installed-SDK preference change. No repair is owed or invented here.
   Preserve PQ869's injected-runtime selection; replace the candidate pin
   with the final reviewed API commit when provisioning the scoped test
   environment.

Owed from the checkpoint itself: pq870 venv provisioning and the
produced-render validation re-run against the updated PB API once root
gives the qualified pin. Do not scan old fleet generations for an
undeployed API.
