# panostitch (browser)

Three.js + React dual-fisheye 360 stitching. See the [repo README](../README.md) for the full story, calibration findings, and the Python library.

```bash
npm install panostitch
```

```jsx
import { OsvPlayer } from "panostitch";
import rig from "./calib/rig.json";
<OsvPlayer rig={rig} /* texA, texB from your OSV decoder */ />
```
