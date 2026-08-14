# Examples

- **`scheme_template.csv`** — the two columns a scheme file must contain.
  `Droplet` holds the ID shown in `droplet_schema.png`, `Group` the treatment
  label. `.xlsx` works just as well; the column names are what matters.

- **`group_map.json`** — optional mapping for `05_group_statistics.py`. Group
  labels typed by hand drift over a project ("Asp 5", "asp  5", "ASP 5"). Keys
  are matched lower-cased and whitespace-collapsed, so one entry covers all
  spellings. Labels missing from the map are reported and excluded rather than
  silently forming their own group.
