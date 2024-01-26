# Testdata
This directory contains different simulation cases with input file and results.
A particular input file should produce a similar (not necessarily exacty equal
due to GPU scheduling "randomness" and atomic floating point operations) result file.

These can be used to verify and test the behaviour of the program.

Build a test case for each input file:
from inside the test function call cubble::run()

Then the entire test battery can be run on triton. Mix of unit tests and larger full simulation tests with comparison of results.

## TODO
- Pitää tallettaa testiajojen data myös, jotta virheen tapahtuessa voi verrata käsin.
- Erilliset testit datojen muodolle? Siis testataan onko about saman verran simuaskeleita?
- Verrataanko testissä yksittäisten rivien samankaltaisuutta vai lasketaanko jotain tilastoja
tuloksista ja verrataan niitä?
- Cli:n API: ottaa sisään inputin ja outputin nimen.
- Pitäisikö output olla myös jsonina? Helppo parsia, voisi myös sisältää inputin.
Tällöin voisi pelkällä resultilla ajaa uuden simun.

```json
{
    input: {
        sisältää koko inputobjektin kaikkine fieldeineen, jotta tämän tiedoston input on sopiva input simulle
    },
    output: {
        jokaiselle outputparamterille oma array:
        snapshots: "some_archive?"
        radius: [],
        phi: [],
        jne: []
    }
}
```
