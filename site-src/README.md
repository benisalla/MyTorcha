# Website source

`docs/` (the published site) is generated from here. Don't edit `docs/*.html` by hand.

```bash
python site-src/gen_ref.py   # regenerate the API reference from the installed package
python site-src/build.py     # wrap pages/*.html in the shared layout -> docs/
python -m http.server 8000 -d docs
```
