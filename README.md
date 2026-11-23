# Continual Learning orvosi képosztályozási feladatok között
**Tantárgy:** Fejlett gépi tanuló modellek  
**Kutatási téma:** 4. feladat – Catastrophic forgetting és task-order hatás vizsgálata orvosi képosztályozáson

---

## 📌 Projekt áttekintése és célkitűzés

A folyamatos tanulás (Continual Learning) célja, hogy a mély tanuló modellek új feladatokat sajátítsanak el egymás után úgy, hogy a korábban megszerzett tudásukat minél kevésbé veszítsék el. Az orvosi diagnosztikában ez kiemelten fontos, hiszen az új intézmények, modalitások és képalkotó eljárások fokozatosan jelennek meg a rendszerekben.

A kutatás fő kérdései:
* Mekkora katasztrofális felejtés (*catastrophic forgetting*) jelentkezik szekvenciális finomhangolás (*Sequential fine-tuning*) során?
* Hogyan befolyásolja a feladatok sorrendje (*task order*) és az adathalmazok közötti reprezentációs hasonlóság (*task similarity*) a végső átlagos pontosságot és a felejtés mértékét?
* Mely continual learning stratégiák (pl. Replay, EWC, paraméter-izoláció) képesek leghatékonyabban enyhíteni a tudásvesztést adott memória- és számítási korlátok mellett?

---

## 🩺 Adathalmazok (MedMNIST 2D)

A kísérletekben a szabványosított **MedMNIST v2** 2D orvosi képosztályozási feladatait használjuk a hivatalos train / validation / test partíciók mentén:

1. **PathMNIST** (9 osztály, szöveti patológia)
2. **DermaMNIST** (7 osztály, bőrgyógyászati elváltozások)
3. **OCTMNIST** (4 osztály, szemészeti retinális rétegfelvételek)
4. **PneumoniaMNIST** (2 osztály, mellkas-röntgen tüdőgyulladás szűrés)
5. **OrganAMNIST** (11 osztály, hasi CT axiális nézet)

---

## 🧠 Modellarchitektúra és Kísérleti Beállítások

* **Beállítás:** Task-incremental Continual Learning (a feladat azonosítója mind a tanítás, mind az értékelés/inferencia közben ismert).
* **Közös gerinchálózat (Backbone):** ResNet-18 architektúra, véletlen súlyinicializálással (`weights=None`), hogy tiszta referenciát biztosítson.
* **Kimeneti rétegek:** Multi-head struktúra feladatspecifikus lineáris fejekkel (`nn.ModuleDict`), amelyek az adott task osztályszámához igazodnak.
* **Kísérleti protokoll:** Minden egyes feladat betanítását követően a modell azonnal kiértékelésre kerül az összes feladat tesztkészletén, felépítve a teljes $T \times T$-s teljesítménymátrixot.

---

## 📐 Kiértékelési Metrikák

A kísérletek lezárásakor az $R \in \mathbb{R}^{T \times T}$ teljesítménymátrixból az alábbi kötelező continual learning mutatókat számítjuk ki és logoljuk:

* **Végső átlagos pontosság ($A_T$):** Az utolsó fázis után az összes feladaton mért tesztpontosság számtani átlaga.
* **Lépésenkénti átlagos pontosság ($A_i$):** Az $i$-edik feladat lefutása után az addig látott feladatok átlagos teljesítménye.
* **Visszamenőleges transzfer (Backward Transfer – BWT):** A korábban megtanult feladatok pontosságvesztésének mértéke a folyamat legvégére a frissen betanított csúcsértékhez képest.
* **Átlagos felejtés (Average Forgetting – $F$):** A tanítás során elért maximális pontosság és a legvégső pontosság különbsége feladatonként átlagolva.
* **Előreirányuló transzfer (Forward Transfer – FWT):** A még nem tanult feladatokon elért teljesítmény mind a nulláról betanított izolált modellekhez (*Single-task oracle*), mind a tanítás előtti *zero-shot* értékekhez viszonyítva.

## 🔬 3. Munkacsomag – Continual Learning módszerek

A `main.ipynb` 3. munkacsomag fejezete a felejtést aktívan csökkentő eljárásokat implementálja. Mind az öt continual ág egyetlen közös futtatókereten (`run_continual_experiment`) osztozik, így azonos kiértékelési protokoll mellett közvetlenül összehasonlíthatók.

| Módszer | Osztály / függvény | Erőforrásköltség |
|---|---|---|
| Sequential fine-tuning | `run_continual_experiment(method="sequential")` | – |
| Replay / exemplar memory | `ExemplarBuffer` (class-balanced, fix 2000 mintás keret) | memória |
| EWC | `EWC` (diagonális Fisher, online akkumuláció, λ = 5000) | plaszticitás |
| Partial freezing | `AdapterResNet18(use_adapters=False)` + `freeze_backbone` | – |
| Adapterek | `AdapterResNet18(use_adapters=True)`, zero-init 1×1 conv | paraméter (~3,1 % / feladat) |
| Joint training (felső referencia) | `run_joint_training` (round-robin batching) | – |

**Protokoll:** eredeti task sorrend, `seed = 42`, azonos optimizer és learning rate minden ágon. A validációs és teszt `DataLoader`-ek itt `drop_last=False` beállítással futnak, ezért a 3. munkacsomag `sequential` ága újra lefuttatja a baseline-t; a WP1 `sequential_baseline` számai nem vethetők össze közvetlenül az itteniekkel.

**Futásonként mentett kimenetek** a `results/` mappába:

```text
wp3_<módszer>_seed42_R_matrix.csv    # nyers T x T mátrix
wp3_<módszer>_seed42_config.json     # teljes konfiguráció + futásnapló (idő, paraméter, memória)
wp3_<módszer>_seed42_metrics.json    # A_T, BWT, felejtés, FWT
wp3_<módszer>_seed42_heatmap.png     # 300 DPI hőtérkép
wp3_joint_seed42_metrics.json        # joint felső referencia
wp3_method_comparison_seed42.csv     # összesítő táblázat (a mentett fájlokból regenerálható)
wp3_method_comparison_seed42.png     # pontosság és felejtés módszerenként
```

Az elméleti háttér és a kódban elkészült részek részletes leírása: `notes/3_munkacsomag_dokumentacio.pdf`.

---

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
