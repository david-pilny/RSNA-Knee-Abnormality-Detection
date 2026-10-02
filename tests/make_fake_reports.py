"""Fake competition root for testing the report labeller locally: <root>/train.csv with made-up reports.

Usage: python tests/make_fake_reports.py <root> [n_studies] [n_gold]
The reports are synthetic (English and German phrase templates), never competition data.
State per finding: P = meets the threshold, M = below it, A = stated absent, N = not mentioned. Gold label = (state == P).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

PHRASES = {
    "en": {
        "ACL": dict(P="Complete rupture of the anterior cruciate ligament.", M="Mucoid degeneration of the ACL without tear.", A="ACL intact."),
        "MCL": dict(P="Acute grade 3 tear of the medial collateral ligament.", M="Grade 1 sprain of the MCL.", A="Medial collateral ligament intact."),
        "Medial Meniscus": dict(P="Complex tear of the posterior horn of the medial meniscus.", M="Intrasubstance degeneration of the medial meniscus, no tear.", A="Medial meniscus normal."),
        "Lateral Meniscus": dict(P="Radial tear of the lateral meniscus body.", M="Mild mucoid degeneration of the lateral meniscus.", A="Lateral meniscus normal."),
        "Medial OA": dict(P="Full-thickness cartilage loss in the medial compartment with advanced osteoarthritis.", M="Mild grade 1 chondropathy of the medial femoral condyle.", A="Cartilage of the medial compartment preserved."),
        "Lateral OA": dict(P="Grade 4 chondropathy of the lateral tibial plateau over a large area.", M="Superficial fibrillation of the lateral compartment cartilage.", A="Lateral compartment cartilage normal."),
        "PF OA": dict(P="Severe patellofemoral osteoarthritis with full-thickness cartilage loss of the patella.", M="Grade 1 chondromalacia patellae.", A="Patellofemoral cartilage intact."),
        "Effusion": dict(P="Large joint effusion.", M="Trace physiological joint fluid.", A="No joint effusion."),
        "Synovitis": dict(P="Marked synovial thickening consistent with synovitis.", M="Minimal synovitis.", A="No synovitis."),
        "Baker's": dict(P="Large Baker's cyst.", M="Tiny Baker's cyst.", A="No Baker's cyst."),
        "Contusion": dict(P="Bone bruise of the lateral femoral condyle.", M="Degenerative subchondral edema.", A="Normal bone marrow signal."),
        "Fracture": dict(P="Acute impaction fracture of the lateral tibial plateau.", M="Old healed fracture deformity.", A="No fracture."),
    },
    "de": {
        "ACL": dict(P="Komplette Ruptur des vorderen Kreuzbandes.", M="Mukoide Degeneration des vorderen Kreuzbandes ohne Riss.", A="Vorderes Kreuzband intakt."),
        "MCL": dict(P="Akute drittgradige Ruptur des medialen Kollateralbandes.", M="Erstgradige Zerrung des Innenbandes.", A="Innenband intakt."),
        "Medial Meniscus": dict(P="Komplexer Riss des Innenmeniskushinterhorns.", M="Intrasubstanzielle Degeneration des Innenmeniskus, kein Riss.", A="Innenmeniskus unauffällig."),
        "Lateral Meniscus": dict(P="Radiärer Riss des Außenmeniskus.", M="Geringe mukoide Degeneration des Außenmeniskus.", A="Außenmeniskus unauffällig."),
        "Medial OA": dict(P="Vollschichtiger Knorpelverlust im medialen Kompartiment bei fortgeschrittener Gonarthrose.", M="Geringe Chondropathie Grad I am medialen Femurkondylus.", A="Knorpel im medialen Kompartiment erhalten."),
        "Lateral OA": dict(P="Chondropathie Grad IV am lateralen Tibiaplateau, großflächig.", M="Oberflächliche Auffaserung des Knorpels lateral.", A="Knorpel im lateralen Kompartiment regelrecht."),
        "PF OA": dict(P="Schwere Retropatellararthrose mit vollschichtigem Knorpelverlust.", M="Chondromalacia patellae Grad I.", A="Retropatellarer Knorpel intakt."),
        "Effusion": dict(P="Ausgeprägter Gelenkerguss.", M="Minimaler, physiologischer Gelenkerguss.", A="Kein Gelenkerguss."),
        "Synovitis": dict(P="Deutliche Synovialverdickung im Sinne einer Synovitis.", M="Minimale Synovitis.", A="Keine Synovitis."),
        "Baker's": dict(P="Große Baker-Zyste.", M="Winzige Baker-Zyste.", A="Keine Baker-Zyste."),
        "Contusion": dict(P="Bone bruise am lateralen Femurkondylus.", M="Degeneratives subchondrales Ödem.", A="Regelrechtes Knochenmarksignal."),
        "Fracture": dict(P="Akute Impressionsfraktur des lateralen Tibiaplateaus.", M="Alte verheilte Frakturdeformität.", A="Keine Fraktur."),
    },
}
INTRO = {"en": "MRI of the knee.", "de": "MRT des Kniegelenks."}


def make(root, n=48, n_gold=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lang = "en" if i % 2 == 0 else "de"
        states = rng.choice(list("PMAN"), size=len(LABELS), p=[0.3, 0.2, 0.3, 0.2])
        text = " ".join([INTRO[lang]] + [PHRASES[lang][l][s] for l, s in zip(LABELS, states) if s != "N"])
        row = {"StudyInstanceUID": f"1.2.3.{1000 + i}", "Report": text}
        for l, s in zip(LABELS, states):
            row[l] = float(s == "P") if i < n_gold else np.nan
        row["_states"] = "".join(states)          # ground truth of the fake data (not in the real train.csv)
        rows.append(row)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    return root / "train.csv"


if __name__ == "__main__":
    args = sys.argv[1:]
    print(make(args[0], *(int(a) for a in args[1:])))
