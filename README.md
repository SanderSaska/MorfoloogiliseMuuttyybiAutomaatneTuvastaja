# Morfoloogilise Muuttüübi Automaatne Tuvastaja

#### Kirjeldus

Tegemist on Kerase mudelitega, mis ennustavad sõnadele muuttüüpe toetudes Väikese Vormisõnastiku[^1] (VVS) tüübikirjeldustele. On loodud kaks LSTM-põhist mudelit: **Sõnamudel** ja **Sõnaliigiga Sõnamudel**.

Sõnamudel on ebatäpsem muuttüübi ennustamises. Mudel oskab ennustada õiget muuttüüpi täpsusega $95.8\%$, kuid muutumatude sõnadega jääb hätta (määrsõnadele määrati õige muuttüüp ainult $14.3\%$).

Sõnaliigiga sõnamudel on võimeline ennustama ka muutumatudele sõnadele õiget muuttüüpi täpsusega $95.2\%$. Üldine täpsus on $97.8\%$. Sõnaliigiga sõnamudel aga eeldab, et lisaks sõnale on kaasa antud ka üldisem sõnaliik -- käändsõna, pöördsõna või muutumatu sõna. Sõnaliik tuleb määrata kasutajal endal.

#### Kasutamine

Kood, millega ennustada muuttüüpe asub III osa muuttüübi tuvastaja *.ipynb* failis alampeatükis "Ennustamine". Kõik koodijupid on vajalikud selles peatükis, et mudelit saaks kasutada.

Funktsioon, millega ennustada muuttüüpi on `leia_muuttüüp(sõna, sõnaliik = '')`.

Sõnamudel võtab sisendiks ainult sõna kirjapildi ehk vaja on määrata ainult `sõna` parameetrit funktsioonis. Sisend peab olema kujul `[sõna_1, sõna_2, ...]` ehk peab koosnema järjendist, mille elemendid on sõnad.

Sõnaliigiga sõnamudel võtab lisaks sõna kirjapildile ka üldisema sõnaliigi. Teisisõnu on vaja määrata mõlemad parameetrid funktsioonis. Üldisem sõnaliik on

* `'n'` - käändsõna;
* `'v'` - pöördsõna;
* `'u'` - muutumatu sõna.

Sisend peab olema kujul `sõna = [sõna_1, sõna_2, ...], sõnaliik = [sõnaliik_1, sõnaliik_2, ...]`.

[^1]: Ü. Viks. Väike vormisõnastik. I: Sissejuhatus & grammatika, II: Sõnastik & lisad. Tallinn: Eesti Teaduste Akadeemia. 1992.