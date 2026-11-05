# Credits

Every sound in `assets/Sounds/` is third-party work downloaded from
[Freesound](https://freesound.org). None of it is covered by this repository's
MIT license: each file keeps the license its author released it under, and this
page carries the attribution those licenses require.

## Audio

| File in this repo | Original title | Author | Source | License |
|---|---|---|---|---|
| `Civil War Field Loop.mp3` | Civil War Field Loop | SergeQuadrado | [freesound.org/s/468542](https://freesound.org/s/468542/) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `Astronomical Explosion.mp3` | Astronomical Explosion | SamsterBirdies | [freesound.org/s/760509](https://freesound.org/s/760509/) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `Sword Fight.mp3` | Sword Fight 1.MP3 | FunWithSound | [freesound.org/s/361483](https://freesound.org/s/361483/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `Cinematic Sound BRAAAM.mp3` | Cinematic Sound BRAAAM Sample — @chakong.audio /// Brass - Trailer | CHAKONG | [freesound.org/s/722633](https://freesound.org/s/722633/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `Victory.mp3` | Victory Music Cue.wav | joshuaempyre | [freesound.org/s/404025](https://freesound.org/s/404025/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `Defeat.mp3` | Piano Game Over Theme (85 BPM) (F Minor) | kanaizo | [freesound.org/s/731657](https://freesound.org/s/731657/) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

### Changes made

CC BY and CC BY-NC ask that modifications be stated. The files were renamed for
use in this project, and one was converted:

- `Sword Fight.mp3` — renamed from *Sword Fight 1.MP3*
- `Cinematic Sound BRAAAM.mp3` — renamed from *Cinematic Sound BRAAAM Sample — @chakong.audio /// Brass - Trailer*
- `Victory.mp3` — renamed from *Victory Music Cue.wav* and converted from WAV to MP3
- `Defeat.mp3` — renamed from *Piano Game Over Theme (85 BPM) (F Minor)*
- `Civil War Field Loop.mp3` and `Astronomical Explosion.mp3` — unchanged

### What each license asks for

- **[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)** — the author
  waived their rights; no conditions attached. Credit is given here anyway, as a
  courtesy.
- **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** — credit the
  author, link the source and the license, and state any changes. All four are
  above.
- **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** —
  everything CC BY asks, **plus no commercial use**.

## Commercial use

`Civil War Field Loop.mp3` — the main theme — is **NonCommercial**. The MIT
license on the code does not extend to it and cannot override it, so while that
file is part of the repository the project as a whole cannot be sold or used
commercially. Sharing it freely, as this project does, is fine.

Swapping it for a CC0 or CC BY track would lift the restriction. The code refers
to it in exactly one place — the `tema` entry in `SUONI`, at the top of
[`frontend/battle/audio/audio.js`](frontend/battle/audio/audio.js) — so replacing
the file and that one line is the whole job.

## JavaScript libraries

| File in this repo | Library | Version | Source | License |
|---|---|---|---|---|
| `frontend/vendor/chart.umd.min.js` | [Chart.js](https://www.chartjs.org) | 4.6.1 | [npm: chart.js](https://www.npmjs.com/package/chart.js) | [MIT](frontend/vendor/chart.js-LICENSE.md) |

The file is the unmodified UMD build from the npm package, copyright banner
included. It ships with the game instead of being loaded from a CDN, so the
radar charts and the simulator keep working without an internet connection.

## Everything else

`assets/icon.ico` and the rest of the repository are not covered by this page.
The code is under the [MIT license](LICENSE).
