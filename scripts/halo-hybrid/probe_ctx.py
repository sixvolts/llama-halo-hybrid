import json,urllib.request,random,sys
random.seed(7); subj=["the harbour","a lighthouse","the old mill","the river ferry","a market square"]; verb=["was rebuilt","stood empty","changed hands","flooded twice","gained a roof"]
def doc(n): return "\n".join(f"Record {i}: {random.choice(subj)} {random.choice(verb)} in {random.randint(1600,1989)}, according to the ledger kept by the clerk." for i in range(n))
def req(prompt, n=96, temp=0):
    body=json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":n,"temperature":temp,"cache_prompt":False}).encode()
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", body, {"Content-Type":"application/json"}), timeout=7200))
    t=d["timings"]; acc=(t.get("draft_n_accepted",0)/t["draft_n"]) if t.get("draft_n") else float('nan'); return t, acc
for n in [int(x) for x in sys.argv[1].split(",")]:
    t,acc=req(doc(n)+"\n\nSummarise the records in two sentences.")
    # with the MTP draft, t/s = tokens per step / step period; tokens per step follows the draft acceptance of the text
    # that happened to be generated, so configs are compared on the step period (ms/step), not on t/s alone
    nmax = int(__import__("os").environ.get("NMAX", "2")); dn = t.get("draft_n", 0)
    steps = dn / nmax if dn else t["predicted_n"]
    print("  %6d records -> prompt %6d tok: prefill %4.0f t/s (%5.1f s), decode %5.2f t/s, acc %.2f n=%d/%d, %.2f tok/step, %.1f ms/step" % (n, t["prompt_n"], t["prompt_per_second"], t["prompt_ms"]/1000, t["predicted_per_second"], acc, t.get("draft_n_accepted",0), dn, t["predicted_n"]/steps, t["predicted_ms"]/steps), flush=True)
