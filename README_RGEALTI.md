# RGEALTI Link Extractor

Scripts to extract download links for RGEALTI (French elevation data) files from IGN geoservices.

## Files

- `extract_rgealti_links.py` - Automated script (requires network access)
- `extract_rgealti_from_files.py` - Manual method using saved HTML files
- `rgealti_links.txt` - Output file containing extracted links

## Method 1: Automated (May be blocked by server)

```bash
python3 extract_rgealti_links.py
```

This script automatically fetches and parses the following pages:
- https://geoservices.ign.fr/telechargement-api/RGEALTI?page=0
- https://geoservices.ign.fr/telechargement-api/RGEALTI?page=1
- https://geoservices.ign.fr/telechargement-api/RGEALTI?page=2
- https://geoservices.ign.fr/telechargement-api/RGEALTI?page=3
- https://geoservices.ign.fr/telechargement-api/RGEALTI?page=4

**Note:** The server may block automated requests with 403 Forbidden errors. If this happens, use Method 2.

## Method 2: Manual HTML Download (Recommended)

1. **Download the HTML pages:**
   - Visit each URL in your browser:
     - https://geoservices.ign.fr/telechargement-api/RGEALTI?page=0
     - https://geoservices.ign.fr/telechargement-api/RGEALTI?page=1
     - https://geoservices.ign.fr/telechargement-api/RGEALTI?page=2
     - https://geoservices.ign.fr/telechargement-api/RGEALTI?page=3
     - https://geoservices.ign.fr/telechargement-api/RGEALTI?page=4

   - For each page: Right-click → "Save Page As" → save as `page0.html`, `page1.html`, etc.

2. **Run the extraction script:**
   ```bash
   python3 extract_rgealti_from_files.py
   ```

3. **View results:**
   ```bash
   cat rgealti_links.txt
   ```

## Example Link Format

The extracted links will look like:
```
https://data.geopf.fr/telechargement/download/RGEALTI/RGEALTI_2-0_1M_ASC_LAMB93-IGN69_D001_2023-08-08/RGEALTI_2-0_1M_ASC_LAMB93-IGN69_D001_2023-08-08.7z.001
```

## Dependencies

```bash
pip3 install beautifulsoup4 requests
```

## Output

- `rgealti_links.txt` - Text file with one download URL per line
- Links are sorted and deduplicated
- Can be used with wget, curl, or download managers

## Download All Files

Once you have the links file, you can download all files using:

```bash
# Using wget
wget -i rgealti_links.txt

# Using curl
cat rgealti_links.txt | xargs -n 1 curl -O

# Using aria2c (faster, parallel downloads)
aria2c -i rgealti_links.txt -j 5
```
