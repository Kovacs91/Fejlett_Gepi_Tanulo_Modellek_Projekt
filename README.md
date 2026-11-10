# Continual Learning orvosi képosztályozási feladatok között
**Tantárgy:** Fejlett gépi tanuló modellek  
**Kutatási téma:** 4. feladat – Catastrophic forgetting és task-order hatás vizsgálata orvosi képosztályozáson

---

## 📌 Projekt áttekintése és célkitűzés

A folyamatos tanulás (Continual Learning) célja, hogy a mély tanuló modellek új feladatokat sajátítsanak el egymás után úgy, hogy a korábban megszerzett tudásukat minél kevésbé veszítsék el[cite: 5]. Az orvosi diagnosztikában ez kiemelten fontos, hiszen az új intézmények, modalitások és képalkotó eljárások fokozatosan jelennek meg a rendszerekben[cite: 5].

A kutatás fő kérdései[cite: 5]:
* Mekkora katasztrofális felejtés (*catastrophic forgetting*) jelentkezik szekvenciális finomhangolás (*Sequential fine-tuning*) során[cite: 5]?
* Hogyan befolyásolja a feladatok sorrendje (*task order*) és az adathalmazok közötti reprezentációs hasonlóság (*task similarity*) a végső átlagos pontosságot és a felejtés mértékét[cite: 5]?
* Mely continual learning stratégiák (pl. Replay, EWC, paraméter-izoláció) képesek leghatékonyabban enyhíteni a tudásvesztést adott memória- és számítási korlátok mellett[cite: 5]?

---

## 🩺 Adathalmazok (MedMNIST 2D)

A kísérletekben a szabványosított **MedMNIST v2** 2D orvosi képosztályozási feladatait használjuk a hivatalos train / validation / test partíciók mentén[cite: 5]:

1. **PathMNIST** (9 osztály, szöveti patológia)[cite: 4, 6]
2. **DermaMNIST** (7 osztály, bőrgyógyászati elváltozások)[cite: 4, 6]
3. **OCTMNIST** (4 osztály, szemészeti retinális rétegfelvételek)[cite: 4, 6]
4. **PneumoniaMNIST** (2 osztály, mellkas-röntgen tüdőgyulladás szűrés)[cite: 4, 6]
5. **OrganAMNIST** (11 osztály, hasi CT axiális nézet)[cite: 4, 6]

---

## 🧠 Modellarchitektúra és Kísérleti Beállítások

* **Beállítás:** Task-incremental Continual Learning (a feladat azonosítója mind a tanítás, mind az értékelés/inferencia közben ismert)[cite: 5, 9].
* **Közös gerinchálózat (Backbone):** ResNet-18 architektúra, véletlen súlyinicializálással (`weights=None`), hogy tiszta referenciát biztosítson[cite: 6].
* **Kimeneti rétegek:** Multi-head struktúra feladatspecifikus lineáris fejekkel (`nn.ModuleDict`), amelyek az adott task osztályszámához igazodnak[cite: 6, 9].
* **Kísérleti protokoll:** Minden egyes feladat betanítását követően a modell azonnal kiértékelésre kerül az összes feladat tesztkészletén, felépítve a teljes $T \times T$-s teljesítménymátrixot[cite: 4, 5].

---

## 📐 Kiértékelési Metrikák

A kísérletek lezárásakor az $R \in \mathbb{R}^{T \times T}$ teljesítménymátrixból az alábbi kötelező continual learning mutatókat számítjuk ki és logoljuk[cite: 4, 5]:

* **Végső átlagos pontosság ($A_T$):** Az utolsó fázis után az összes feladaton mért tesztpontosság számtani átlaga.
* **Lépésenkénti átlagos pontosság ($A_i$):** Az $i$-edik feladat lefutása után az addig látott feladatok átlagos teljesítménye[cite: 4].
* **Visszamenőleges transzfer (Backward Transfer – BWT):** A korábban megtanult feladatok pontosságvesztésének mértéke a folyamat legvégére a frissen betanított csúcsértékhez képest[cite: 4].
* **Átlagos felejtés (Average Forgetting – $F$):** A tanítás során elért maximális pontosság és a legvégső pontosság különbsége feladatonként átlagolva[cite: 4].
* **Előreirányuló transzfer (Forward Transfer – FWT):** A még nem tanult feladatokon elért teljesítmény mind a nulláról betanított izolált modellekhez (*Single-task oracle*), mind a tanítás előtti *zero-shot* értékekhez viszonyítva[cite: 4].

## 👥 Csapat
**Ugrai Gergő**

**Csorba Csanád**

**Kovács Dávid Attila**

---

## 📁 Repository Mappastruktúra

```text
├── .gitignore                          # Kizárt segédfájlok (.idea, cache, modellek)
├── README.md                           # Projekt áttekintés és dokumentáció
├── 04_continual_learning_task_order.pdf# Hivatalos kutatási feladatkiírás
├── main.ipynb                          # Fő kutatási notebook (pipeline, tanítás, metrikák)
├── notes/                              # Beszkennelt kézzel írott kutatási feljegyzések
└── results/                            # Géppel olvasható és vizuális kísérleti kimenetek
    ├── sequential_baseline_seed42.json # Részletes metrikák strukturált JSON riportban
    ├── sequential_baseline_seed42.csv  # Mátrix és lépésenkénti pontosságok táblázata
    ├── single_task_reference.json      # Izolált Single-Task csúcsteljesítmények
    └── sequential_baseline_seed42_heatmap.png # 300 DPI publikációs hőtérkép
