"""
Lurgi Methanol Plant — Techno-Economic Analysis (TEA) Script
=============================================================
Multi-case analysis: 3 capacity scales × 3 price scenarios

CAPACITY CASES
──────────────
  Case 1 (Base)  : 549 kmol/hr feed
  Case 2 (1.5×)  : 823.5 kmol/hr feed
  Case 3 (2×)    : 1098 kmol/hr feed

Scale-up uses the 0.6 power law (six-tenths rule) for equipment costs.
Mass/energy flows scale linearly with feed rate.
Plant capacity reported in TPD (tonnes/day, 24 hr basis).

PRICE SCENARIOS
───────────────
  Low   : syngas $0.08/Nm³, MeOH $320/tonne
  Base  : syngas $0.12/Nm³, MeOH $400/tonne
  High  : syngas $0.18/Nm³, MeOH $480/tonne
"""

import win32com.client
import time, os, math

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ASPEN_FILE = r"E:\Lurgi Methanol Process 1 Feed With Utilities.bkp"

BASE_FLOW_KMOLHR = 549.0        # kmol/hr — base case feed rate

SCALE_FACTORS = [1.0, 1.5, 2.0] # capacity multipliers
SCALE_LABELS  = ["Case 1 (549 kmol/hr)", "Case 2 (1.5×)", "Case 3 (2×)"]

PRICE_SCENARIOS = {
    "Low  prices": {"syngas": 0.08, "meoh": 320, "co2_tax": 30},
    "Base prices": {"syngas": 0.12, "meoh": 400, "co2_tax": 50},
    "High prices": {"syngas": 0.18, "meoh": 480, "co2_tax": 75},
}

PLANT_CONFIG = {
    "operating_hours_per_year": 8000,
    "operating_days_per_year":  330,   # for TPD reporting
    "plant_lifetime_years":     20,
    "discount_rate":            0.10,
    "tax_rate":                 0.25,
    "lang_factor":              4.74,
    "indirect_cost_factor":     0.30,
    "working_capital_months":   1.0,
    "maintenance_pct_fci":      0.03,
    "insurance_pct_fci":        0.01,
    "labour_operators":         15,
    "labour_wage_usd_per_yr":   70_000,
    "overhead_pct_labour":      0.50,
    "cepci_base":               567.3,
    "cepci_target":             816.0,
    "scale_exponent":           0.60,  # six-tenths power law
}

MARKET_BASE = {
    "syngas_usd_per_nm3":            0.12,
    "meoh_usd_per_tonne":            400.0,
    "byproduct_credit_usd_per_tonne": 50.0,
    "co2_tax_usd_per_tonne":          50.0,
    "hps_usd_per_tonne":              18.0,
    "lps_usd_per_tonne":              12.0,
    "mps_usd_per_tonne":              15.0,
    "mps_gen_credit_usd_per_tonne":   14.0,
    "power_usd_per_kwh":               0.07,
    "tcw_usd_per_tonne":               0.25,
    "air_cool_usd_per_gj":             2.0,
}

PLANT_SECTIONS = {
    "Feed Compression":      ["K201-S1", "K201-S2", "K202"],
    "Feed Preheating":       ["E202-HX", "E202-C", "E202-H"],
    "Reaction":              ["R201-A", "R201-B", "R201-C", "R201-D"],
    "Phase Separation":      ["V201", "V208", "T-351"],
    "Recycle Compression":   ["K203", "K204"],
    "Crude MeOH Column":     ["C201"],
    "MeOH Purification":     ["C-301", "C302"],
    "Heat Recovery":         ["E301-C", "E301-H", "HX-E301", "E203",
                               "E206", "E207", "E209", "E210"],
    "Product Handling":      ["P301", "P302", "P304"],
}

EQUIPMENT_PARAMS = {
    "compressor":          {"Cb": 580_000, "Sb": 1000,  "n": 0.67, "unit": "kW"},
    "heat_exchanger":      {"Cb":  35_000, "Sb":  100,  "n": 0.60, "unit": "m²"},
    "distillation_column": {"Cb": 130_000, "Sb":   50,  "n": 0.78, "unit": "trays"},
    "vessel":              {"Cb":  60_000, "Sb":   10,  "n": 0.65, "unit": "m³"},
    "pump":                {"Cb":   6_500, "Sb":   10,  "n": 0.55, "unit": "kW"},
    "reactor":             {"Cb": 750_000, "Sb":  100,  "n": 0.60, "unit": "m³"},
}

BLOCK_EQUIPMENT = {
    "K201-S1": {"type":"compressor",         "proxy":"power_kw",  "fixed_S":None},
    "K201-S2": {"type":"compressor",         "proxy":"power_kw",  "fixed_S":None},
    "K202":    {"type":"compressor",         "proxy":"power_kw",  "fixed_S":None},
    "E202-HX": {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E202-C":  {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E202-H":  {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "R201-A":  {"type":"reactor",            "proxy":"fixed",     "fixed_S":150},
    "R201-B":  {"type":"reactor",            "proxy":"fixed",     "fixed_S":150},
    "R201-C":  {"type":"reactor",            "proxy":"fixed",     "fixed_S":150},
    "R201-D":  {"type":"reactor",            "proxy":"fixed",     "fixed_S":150},
    "V201":    {"type":"vessel",             "proxy":"fixed",     "fixed_S":25},
    "V208":    {"type":"vessel",             "proxy":"fixed",     "fixed_S":15},
    "T-351":   {"type":"vessel",             "proxy":"fixed",     "fixed_S":20},
    "K203":    {"type":"compressor",         "proxy":"power_kw",  "fixed_S":None},
    "K204":    {"type":"compressor",         "proxy":"power_kw",  "fixed_S":None},
    "C201":    {"type":"distillation_column","proxy":"fixed",     "fixed_S":35},
    "C-301":   {"type":"distillation_column","proxy":"fixed",     "fixed_S":60},
    "C302":    {"type":"distillation_column","proxy":"fixed",     "fixed_S":45},
    "E301-C":  {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E301-H":  {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "HX-E301": {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E203":    {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E206":    {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E207":    {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E209":    {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "E210":    {"type":"heat_exchanger",     "proxy":"duty_gj",   "fixed_S":None},
    "P301":    {"type":"pump",               "proxy":"power_kw",  "fixed_S":None},
    "P302":    {"type":"pump",               "proxy":"power_kw",  "fixed_S":None},
    "P304":    {"type":"pump",               "proxy":"power_kw",  "fixed_S":None},
}

FEED_COMP_CANDIDATES = [
    r"\Data\Streams\TOTFEED\Input\FLOW\MIXED",
    r"\Data\Streams\TOTFEED\Input\COMP\FLOW\MIXED",
]

OUTPUT_STREAMS = [
    ("Methanol Product",   "MEOH"),
    ("Light Ends Stream",  "S41"),
    ("By-Products Stream", "S39"),
    ("Water to Treatment", "S44"),
]

UTILITIES = [
    ("Air Cooling",           "AIR-COOL"),
    ("High Pressure Steam",   "HPS"),
    ("Low Pressure Steam",    "LPS"),
    ("Medium Pressure Steam", "MPS"),
    ("MPS Generation",        "MPS-GEN"),
    ("Power",                 "POWER"),
    ("Tempered Cooling Water","TCW"),
]

MEOH_MW       = 32.04   # g/mol
HRS_PER_DAY   = 24.0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ASPEN COM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_find(tree, path):
    try:
        return tree.FindNode(path)
    except Exception:
        return None

def first_scalar(tree, paths):
    for p in paths:
        n = safe_find(tree, p)
        if n is None: continue
        try:
            v = n.Value
            if v is not None: return v, p
        except Exception:
            pass
    return None, None

def first_comp_node(tree, paths):
    for p in paths:
        n = safe_find(tree, p)
        if n is None: continue
        try:
            d = {child.Name: child.Value for child in n.Elements
                 if child.Value is not None}
            if d: return d, p
        except Exception:
            pass
    return {}, None

def sum_children(tree, path):
    n = safe_find(tree, path)
    if n is None: return None
    try:
        vals = [float(child.Value) for child in n.Elements
                if child.Value is not None]
        return sum(vals) if vals else None
    except Exception:
        return None

def s_out(sid, var, phase="MIXED"):
    return rf"\Data\Streams\{sid}\Output\{var}\{phase}"

def s_in(sid, var, phase="MIXED"):
    return rf"\Data\Streams\{sid}\Input\{var}\{phase}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SIMULATION DATA COLLECTION (base case only)
# ══════════════════════════════════════════════════════════════════════════════

def collect_simulation_data(tree):
    data = {"streams": {}, "utilities": {}, "blocks": {}}

    for label, sid in OUTPUT_STREAMS:
        temp,_    = first_scalar(tree, [s_out(sid,"TEMP_OUT"), s_out(sid,"TEMP")])
        pres,_    = first_scalar(tree, [s_out(sid,"PRES_OUT"), s_out(sid,"PRES")])
        flow,_    = first_scalar(tree, [s_out(sid,"MOLEFLMX"), s_out(sid,"MASSFLMX")])
        comps,cp  = first_comp_node(tree, [s_out(sid,"MOLEFLOW"), s_out(sid,"MOLEFRAC"),
                                            s_out(sid,"MASSFLOW"), s_out(sid,"MASSFRAC")])
        data["streams"][sid] = {"label":label,"temp_c":temp,"pres_bar":pres,
                                "flow":flow,"comps":comps,"comp_path":cp}

    for label, uid in UTILITIES:
        duty  = sum_children(tree, rf"\Data\Utilities\{uid}\Output\UTL_DUTY")
        usage = sum_children(tree, rf"\Data\Utilities\{uid}\Output\UTL_USAGE")
        co2   = sum_children(tree, rf"\Data\Utilities\{uid}\Output\BLK_CO2RATE")
        if duty  is None:
            duty,_ = first_scalar(tree, [rf"\Data\Utilities\{uid}\Output\UTL_HCOOL"])
        if usage is None:
            usage,_= first_scalar(tree, [rf"\Data\Utilities\{uid}\Output\UTL_TRATE"])
        data["utilities"][uid] = {"label":label,"duty_gj_hr":duty,
                                   "usage_kg_hr":usage,"co2_tonne_hr":co2}

    for bid, info in BLOCK_EQUIPMENT.items():
        duty_val = power_val = None
        if info["proxy"] == "duty_gj":
            duty_val,_ = first_scalar(tree, [
                rf"\Data\Blocks\{bid}\Output\QNET",
                rf"\Data\Blocks\{bid}\Output\DUTY"])
        elif info["proxy"] == "power_kw":
            power_val,_ = first_scalar(tree, [
                rf"\Data\Blocks\{bid}\Output\WNET",
                rf"\Data\Blocks\{bid}\Output\POWER",
                rf"\Data\Blocks\{bid}\Output\BHP"])
        data["blocks"][bid] = {"duty_gj_hr":duty_val,"power_kw":power_val,
                               "fixed_S":info["fixed_S"]}

    totfeed,_ = first_scalar(tree, [r"\Data\Streams\TOTFEED\Input\TOTFLOW\MIXED"])
    data["totfeed_kmol_hr"] = totfeed

    meoh_flow,_ = first_scalar(tree, [s_out("MEOH","MOLEFLMX"), s_out("MEOH","MASSFLMX")])
    data["meoh_moleflow_kmolhr"] = meoh_flow
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CAPEX (supports scale factor)
# ══════════════════════════════════════════════════════════════════════════════

def equipment_purchase_cost_base(bid, block_data, cepci_ratio):
    info   = BLOCK_EQUIPMENT[bid]
    params = EQUIPMENT_PARAMS[info["type"]]
    Cb, Sb, n = params["Cb"], params["Sb"], params["n"]
    bdata  = block_data.get(bid, {})

    if info["proxy"] == "duty_gj":
        duty = bdata.get("duty_gj_hr")
        S = abs(duty) * 111.1 if (duty and abs(duty) > 0.001) else 50.0
    elif info["proxy"] == "power_kw":
        pwr = bdata.get("power_kw")
        S = abs(pwr) if (pwr and abs(pwr) > 0.1) else 500.0
    else:
        S = bdata.get("fixed_S") or info["fixed_S"] or Sb

    S = max(S, Sb * 0.05)
    S = min(S, Sb * 50.0)
    return Cb * (S / Sb) ** n * cepci_ratio, S, params["unit"]


def calculate_capex(block_data, config, scale_factor=1.0):
    cepci_ratio = config["cepci_target"] / config["cepci_base"]
    lang        = config["lang_factor"]
    indirect    = config["indirect_cost_factor"]
    exp         = config["scale_exponent"]

    section_pce = {s: 0.0 for s in PLANT_SECTIONS}
    eq_detail   = {}
    total_pce   = 0.0

    for section, blocks in PLANT_SECTIONS.items():
        for bid in blocks:
            if bid not in BLOCK_EQUIPMENT: continue
            pce_base, S, unit = equipment_purchase_cost_base(bid, block_data, cepci_ratio)
            pce_scaled = pce_base * (scale_factor ** exp)
            section_pce[section] += pce_scaled
            total_pce            += pce_scaled
            eq_detail[bid] = {"section":section,"type":BLOCK_EQUIPMENT[bid]["type"],
                               "S":S,"unit":unit,"pce_usd":pce_scaled}

    fci       = total_pce * lang
    indirect_c= fci * indirect
    tfc       = fci + indirect_c

    return {"total_pce_usd":total_pce,"fci_usd":fci,"indirect_usd":indirect_c,
            "total_fixed_usd":tfc,"section_pce":section_pce,
            "section_fci":{s:v*lang for s,v in section_pce.items()},
            "eq_detail":eq_detail}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — OPEX (supports scale factor + custom market prices)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_opex(sim_data, config, market, capex, scale_factor=1.0):
    hrs   = config["operating_hours_per_year"]
    m     = market
    utils = sim_data["utilities"]

    totfeed_base  = sim_data.get("totfeed_kmol_hr") or 0.0
    syngas_nm3_hr = totfeed_base * 22.414 * scale_factor
    feed_cost_annual = syngas_nm3_hr * m["syngas_usd_per_nm3"] * hrs

    def utl(uid, key, price_per_t):
        usage = (utils[uid].get("usage_kg_hr") or 0.0) * scale_factor
        return (usage / 1000.0) * price_per_t * hrs

    def power_cost(uid):
        usage = (utils[uid].get("usage_kg_hr") or 0.0) * scale_factor
        kw    = usage / 3600.0
        return kw * m["power_usd_per_kwh"] * hrs

    def air_cost(uid):
        duty = (utils[uid].get("duty_gj_hr") or 0.0) * scale_factor
        return abs(duty) * m["air_cool_usd_per_gj"] * hrs

    util_costs = {
        "AIR-COOL": air_cost("AIR-COOL"),
        "HPS":      utl("HPS",  "usage_kg_hr", m["hps_usd_per_tonne"]),
        "LPS":      utl("LPS",  "usage_kg_hr", m["lps_usd_per_tonne"]),
        "MPS":      utl("MPS",  "usage_kg_hr", m["mps_usd_per_tonne"]),
        "MPS-GEN":  -utl("MPS-GEN","usage_kg_hr", m["mps_gen_credit_usd_per_tonne"]),
        "POWER":    power_cost("POWER"),
        "TCW":      utl("TCW",  "usage_kg_hr", m["tcw_usd_per_tonne"]),
    }
    total_util = sum(util_costs.values())

    co2_total_hr = sum((v.get("co2_tonne_hr") or 0) for v in utils.values()) * scale_factor
    co2_tax_annual = co2_total_hr * m["co2_tax_usd_per_tonne"] * hrs

    fci         = capex["total_fixed_usd"]
    maintenance = fci * config["maintenance_pct_fci"]
    insurance   = fci * config["insurance_pct_fci"]
    labour      = config["labour_operators"] * config["labour_wage_usd_per_yr"]
    overheads   = labour * config["overhead_pct_labour"]
    total_fixed = maintenance + insurance + labour + overheads

    total_opex  = feed_cost_annual + total_util + co2_tax_annual + total_fixed

    return {"feed_cost_annual":feed_cost_annual,"syngas_nm3_hr":syngas_nm3_hr,
            "util_costs":util_costs,"total_util":total_util,
            "co2_tax_annual":co2_tax_annual,"co2_total_hr":co2_total_hr,
            "maintenance":maintenance,"labour":labour+overheads,
            "insurance":insurance,"total_fixed":total_fixed,
            "total_opex_annual":total_opex}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — REVENUE & PROFITABILITY
# ══════════════════════════════════════════════════════════════════════════════

def calculate_revenue_and_profit(sim_data, config, market, capex, opex, scale_factor):
    hrs  = config["operating_hours_per_year"]
    days = config["operating_days_per_year"]
    m    = market

    meoh_kmolhr  = (sim_data.get("meoh_moleflow_kmolhr") or 0.0) * scale_factor
    meoh_kghr    = meoh_kmolhr * MEOH_MW
    meoh_tonne_yr= (meoh_kghr / 1000.0) * hrs
    meoh_tpd     = (meoh_kghr / 1000.0) * HRS_PER_DAY  # tonne/day (24hr basis)

    meoh_revenue     = meoh_tonne_yr * m["meoh_usd_per_tonne"]
    byproduct_revenue = 0.0
    for sid in ["S39","S41"]:
        flow = (sim_data["streams"].get(sid,{}).get("flow") or 0.0) * scale_factor
        byproduct_revenue += (flow * 30.0 / 1000.0) * hrs * m["byproduct_credit_usd_per_tonne"]
    total_revenue = meoh_revenue + byproduct_revenue

    tfc   = capex["total_fixed_usd"]
    wc    = opex["feed_cost_annual"] * (config["working_capital_months"] / 12.0)
    tci   = tfc + capex["indirect_usd"] + wc

    total_opex    = opex["total_opex_annual"]
    ebitda        = total_revenue - total_opex
    depreciation  = tfc / config["plant_lifetime_years"]
    ebit          = ebitda - depreciation
    tax           = max(ebit * config["tax_rate"], 0.0)
    net_profit    = ebit - tax
    annual_cf     = net_profit + depreciation

    r   = config["discount_rate"]
    n   = config["plant_lifetime_years"]
    npv = -tci + sum(annual_cf / (1+r)**t for t in range(1,n+1))
    payback = tci / annual_cf if annual_cf > 0 else float("nan")
    irr = _irr([-tci]+[annual_cf]*n)
    capital_charge = tci * r / (1-(1+r)**(-n))
    cop  = total_opex / meoh_tonne_yr if meoh_tonne_yr > 0 else float("nan")
    full_cost = (total_opex + capital_charge) / meoh_tonne_yr if meoh_tonne_yr > 0 else float("nan")

    return {"meoh_kmolhr":meoh_kmolhr,"meoh_tonne_yr":meoh_tonne_yr,"meoh_tpd":meoh_tpd,
            "meoh_revenue":meoh_revenue,"byproduct_revenue":byproduct_revenue,
            "total_revenue":total_revenue,"wc":wc,"tci":tci,
            "ebitda":ebitda,"depreciation":depreciation,"net_profit":net_profit,
            "annual_cf":annual_cf,"npv":npv,"payback":payback,
            "irr_pct":(irr*100 if irr else float("nan")),
            "cop":cop,"full_cost":full_cost,"capital_charge":capital_charge,
            "ebitda_margin":(ebitda/total_revenue*100 if total_revenue>0 else float("nan"))}


def _irr(cf, guess=0.10, tol=1e-6, imax=200):
    r = guess
    for _ in range(imax):
        f  = sum(c/(1+r)**t for t,c in enumerate(cf))
        df = sum(-t*c/(1+r)**(t+1) for t,c in enumerate(cf))
        if abs(df) < 1e-12: break
        nr = r - f/df
        if abs(nr-r) < tol: return nr
        r  = nr
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — RUN ALL CASES
# ══════════════════════════════════════════════════════════════════════════════

def run_all_cases(sim_data):
    """
    Run CAPEX/OPEX/Profitability for 3 scale factors × 3 price scenarios.
    Returns nested dict: results[scale_label][price_scenario]
    """
    results = {}
    for sf, sl in zip(SCALE_FACTORS, SCALE_LABELS):
        results[sl] = {}
        capex = calculate_capex(sim_data["blocks"], PLANT_CONFIG, sf)
        for pname, prices in PRICE_SCENARIOS.items():
            mkt = {**MARKET_BASE,
                   "syngas_usd_per_nm3":  prices["syngas"],
                   "meoh_usd_per_tonne":  prices["meoh"],
                   "co2_tax_usd_per_tonne": prices["co2_tax"]}
            opex   = calculate_opex(sim_data, PLANT_CONFIG, mkt, capex, sf)
            profit = calculate_revenue_and_profit(sim_data, PLANT_CONFIG, mkt, capex, opex, sf)
            results[sl][pname] = {"capex":capex,"opex":opex,"profit":profit,
                                  "market":mkt,"scale_factor":sf}
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "=" * 80
SEP2 = "-" * 80

def _m(v, dec=2):
    if v is None or (isinstance(v,float) and math.isnan(v)): return "N/A"
    return f"${v/1e6:.{dec}f}M"

def _f(v, dec=2, sfx=""):
    if v is None or (isinstance(v,float) and math.isnan(v)): return "N/A"
    return f"{v:.{dec}f}{sfx}"

def section(title):
    print(f"\n{SEP}\n   {title}\n{SEP}")

def display_streams(sim_data):
    section("PRODUCT STREAM RESULTS  (base case — from Aspen simulation)")
    for sid, sd in sim_data["streams"].items():
        print(f"\n  ── {sd['label']}  ({sid}) ──")
        print(f"  Temperature : {_f(sd['temp_c'],4)} °C")
        print(f"  Pressure    : {_f(sd['pres_bar'],6)} bar")
        print(f"  Total flow  : {_f(sd['flow'],4)} kmol/hr")
        comps = sd.get("comps",{})
        if comps:
            active = {k:v for k,v in comps.items() if v and abs(float(v))>1e-10}
            print(f"  {'Component':<14}  {'Value':>20}")
            for k,v in (active or comps).items():
                print(f"  {k:<14}  {v:>20.8f}")

def display_utilities(sim_data):
    section("UTILITY SUMMARY  (base case — from Aspen simulation)")
    print(f"\n  {'Utility':<26}  {'Duty (GJ/hr)':>14}  {'Usage (kg/hr)':>15}  {'CO2 (t/hr)':>12}")
    print(f"  {'-'*26}  {'-'*14}  {'-'*15}  {'-'*12}")
    for uid, ud in sim_data["utilities"].items():
        d = _f(ud["duty_gj_hr"],4)  if ud["duty_gj_hr"]  is not None else "N/A"
        u = _f(ud["usage_kg_hr"],2) if ud["usage_kg_hr"] is not None else "N/A"
        c = _f(ud["co2_tonne_hr"],4)if ud["co2_tonne_hr"]is not None else "—"
        print(f"  {ud['label']:<26}  {d:>14}  {u:>15}  {c:>12}")

def display_capex_multi(results):
    section("CAPITAL EXPENDITURE — 3 CAPACITY CASES")
    print(f"\n  {'':38}  {'Case 1 (Base)':>16}  {'Case 2 (1.5×)':>16}  {'Case 3 (2×)':>16}")
    print(f"  {'Item':<38}  {'TPD':>16}  {'TPD':>16}  {'TPD':>16}")

    cases = list(results.keys())
    base_prices = list(PRICE_SCENARIOS.keys())[1]  # "Base prices"
    rows = [("Methanol production (TPD)",   lambda r: f"{r['profit']['meoh_tpd']:.0f} TPD"),
            ("PCE",                          lambda r: _m(r['capex']['total_pce_usd'])),
            ("FCI",                          lambda r: _m(r['capex']['fci_usd'])),
            ("Total Fixed Capital (TFC)",    lambda r: _m(r['capex']['total_fixed_usd'])),
            ("Working capital",              lambda r: _m(r['profit']['wc'])),
            ("TOTAL CAPITAL INVESTMENT",     lambda r: _m(r['profit']['tci'])),]

    for label, fn in rows:
        vals = [fn(results[c][base_prices]) for c in cases]
        print(f"  {label:<38}  {vals[0]:>16}  {vals[1]:>16}  {vals[2]:>16}")

    print(f"\n  CAPEX by section (FCI, base prices):")
    print(f"  {'Section':<32}  {'Case 1':>12}  {'Case 2':>12}  {'Case 3':>12}")
    print(f"  {'-'*32}  {'-'*12}  {'-'*12}  {'-'*12}")
    for sec in PLANT_SECTIONS:
        vals = [_m(results[c][base_prices]["capex"]["section_fci"].get(sec,0)) for c in cases]
        print(f"  {sec:<32}  {vals[0]:>12}  {vals[1]:>12}  {vals[2]:>12}")

def display_opex_multi(results):
    section("OPERATING EXPENDITURE — 3 CASES × 3 PRICE SCENARIOS")
    for sl in SCALE_LABELS:
        tpd = results[sl][list(PRICE_SCENARIOS.keys())[1]]["profit"]["meoh_tpd"]
        print(f"\n  {sl}  ({tpd:.0f} TPD)")
        print(f"  {'OPEX item':<36}  {'Low prices':>14}  {'Base prices':>14}  {'High prices':>14}")
        print(f"  {'-'*36}  {'-'*14}  {'-'*14}  {'-'*14}")
        items = [
            ("Feedstock (syngas)",     lambda o: o["feed_cost_annual"]),
            ("Total utilities",        lambda o: o["total_util"]),
            ("CO2 emission tax",       lambda o: o["co2_tax_annual"]),
            ("Fixed costs",            lambda o: o["total_fixed"]),
            ("TOTAL ANNUAL OPEX",      lambda o: o["total_opex_annual"]),
        ]
        for label, fn in items:
            vals = [_m(fn(results[sl][p]["opex"])) for p in PRICE_SCENARIOS]
            sep_line = "─"*36+"  "+"─"*14+"  "+"─"*14+"  "+"─"*14
            if label.startswith("TOTAL"):
                print(f"  {sep_line}")
            print(f"  {label:<36}  {vals[0]:>14}  {vals[1]:>14}  {vals[2]:>14}")

def display_profitability_matrix(results):
    section("PROFITABILITY MATRIX — 3 CAPACITY CASES × 3 PRICE SCENARIOS")

    metrics = [
        ("MeOH production (TPD)",       lambda r: _f(r["profit"]["meoh_tpd"],0)),
        ("Total revenue ($/yr)",         lambda r: _m(r["profit"]["total_revenue"])),
        ("Total OPEX ($/yr)",            lambda r: _m(r["profit"]["total_opex_annual"],0)),  # fixed attr name
        ("EBITDA ($/yr)",                lambda r: _m(r["profit"]["ebitda"])),
        ("EBITDA margin (%)",            lambda r: _f(r["profit"]["ebitda_margin"],1)+"%"),
        ("Net profit ($/yr)",            lambda r: _m(r["profit"]["net_profit"])),
        ("CoP — OPEX only ($/t)",        lambda r: "$"+_f(r["profit"]["cop"],2)+"/t"),
        ("CoP — full cost ($/t)",        lambda r: "$"+_f(r["profit"]["full_cost"],2)+"/t"),
        ("TCI",                          lambda r: _m(r["profit"]["tci"])),
        ("NPV",                          lambda r: _m(r["profit"]["npv"])),
        ("IRR (%)",                      lambda r: _f(r["profit"]["irr_pct"],1)+"%"),
        ("Payback (yr)",                 lambda r: _f(r["profit"]["payback"],1)+" yr"),
    ]

    for pname in PRICE_SCENARIOS:
        print(f"\n  Price scenario: {pname}")
        print(f"  {'Metric':<32}  {'Case 1':>16}  {'Case 2':>16}  {'Case 3':>16}")
        print(f"  {'-'*32}  {'-'*16}  {'-'*16}  {'-'*16}")
        for label, fn in metrics:
            # patch opex total attr difference
            def safe_fn(r, fn=fn):
                try: return fn(r)
                except: return "N/A"
            vals = [safe_fn(results[sl][pname]) for sl in SCALE_LABELS]
            print(f"  {label:<32}  {vals[0]:>16}  {vals[1]:>16}  {vals[2]:>16}")

def display_equipment_detail(results):
    section("EQUIPMENT COST DETAIL  (base case, CEPCI-indexed)")
    capex = results[SCALE_LABELS[0]][list(PRICE_SCENARIOS.keys())[1]]["capex"]
    print(f"\n  {'Block':<12}  {'Type':<22}  {'Size':>10}  {'Unit':<6}  {'PCE':>12}")
    print(f"  {'-'*12}  {'-'*22}  {'-'*10}  {'-'*6}  {'-'*12}")
    cur_sec = ""
    for bid, ed in sorted(capex["eq_detail"].items(),
                          key=lambda x: (x[1]["section"], x[0])):
        if ed["section"] != cur_sec:
            cur_sec = ed["section"]
            print(f"\n  [{cur_sec}]")
        print(f"  {bid:<12}  {ed['type']:<22}  {ed['S']:>10.1f}  {ed['unit']:<6}  {_m(ed['pce_usd']):>12}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(ASPEN_FILE):
        raise FileNotFoundError(f"File not found: {ASPEN_FILE}")

    aspen = win32com.client.Dispatch("Apwn.Document")

    try:
        print("\nInitializing Aspen Plus...")
        aspen.InitFromFile(ASPEN_FILE)
        time.sleep(5)
        aspen.Visible = 1
        print("Aspen Plus loaded.\n")

        comp_node, comp_base = None, None
        for p in FEED_COMP_CANDIDATES:
            n = safe_find(aspen.Tree, p)
            if n is not None:
                comp_node, comp_base = n, p
                print(f"Feed composition path: {p}")
                break
        if comp_node is None:
            raise RuntimeError("Cannot locate TOTFEED composition input node.")

        components = [c.Name for c in comp_node.Elements]
        print(f"Components: {', '.join(components)}\n")

        # ── User inputs (base case) ────────────────────────────────────────
        total_flow  = float(input(f"Total feed flowrate — base case (kmol/hr) [{BASE_FLOW_KMOLHR}]: ") or BASE_FLOW_KMOLHR)
        print("Enter component flows (kmol/hr) — base case:")
        user_vals   = {c: float(input(f"  {c}: ") or 0) for c in components}
        temperature = float(input("Feed temperature (°C): "))
        pressure    = float(input("Feed pressure (bar): "))

        n = safe_find(aspen.Tree, r"\Data\Streams\TOTFEED\Input\TOTFLOW\MIXED")
        if n: n.Value = total_flow
        for comp, val in user_vals.items():
            n = safe_find(aspen.Tree, comp_base + "\\" + comp)
            if n: n.Value = val
        n = safe_find(aspen.Tree, r"\Data\Streams\TOTFEED\Input\TEMP\MIXED")
        if n: n.Value = temperature
        n = safe_find(aspen.Tree, r"\Data\Streams\TOTFEED\Input\PRES\MIXED")
        if n: n.Value = pressure

        print("\nRunning simulation (base case)...")
        aspen.Engine.Run2()
        while aspen.Engine.IsRunning:
            time.sleep(1)
        print("Simulation complete.\n")

        print("Collecting simulation results...")
        sim_data = collect_simulation_data(aspen.Tree)

        print("Running multi-case TEA (3 capacities × 3 price scenarios)...")
        all_results = run_all_cases(sim_data)
        print("TEA complete.\n")

        # ── Display ────────────────────────────────────────────────────────
        display_streams(sim_data)
        display_utilities(sim_data)
        display_capex_multi(all_results)
        display_equipment_detail(all_results)
        display_opex_multi(all_results)
        display_profitability_matrix(all_results)

        print(f"\n{SEP}\n  Analysis complete — 3 capacity cases × 3 price scenarios.\n{SEP}\n")

    finally:
        print("Closing Aspen Plus...")
        try: aspen.Close()
        except Exception: pass
        aspen = None
        print("Done.")


if __name__ == "__main__":
    main()
