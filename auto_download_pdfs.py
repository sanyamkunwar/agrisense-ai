import os
import requests
from tqdm import tqdm

DOWNLOAD_DIR = "data/knowledge_base"

# Create folder if not exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ----------------------------------------------------------
#  LIST OF PDF LINKS (Batch 2) – All new ones
# ----------------------------------------------------------
pdf_links = {
    # Tomato
    "Tomato_Diseases_UF.pdf":
        "https://edis.ifas.ufl.edu/pdffiles/PG/PG12400.pdf",
    "Tomato_Production_Arkansas.pdf":
        "https://www.uaex.uada.edu/publications/pdf/mp424.pdf",
    "Tomato_Disease_Diagnosis_Missouri.pdf":
        "https://extension.missouri.edu/media/wysiwyg/Extensiondata/Pub/pdf/agguides/pests/g06070.pdf",

    # Potato
    "Potato_Disease_Management_Idaho.pdf":
        "https://www.uidaho.edu/-/media/uidao-extension/publications/bul/bul0741.pdf",
    "Potato_IPM_WSU.pdf":
        "https://s3.wp.wsu.edu/uploads/sites/2071/2014/05/EM038.pdf",
    "Potato_Production_Handbook_Idaho.pdf":
        "https://www.extension.uidaho.edu/publishing/pdf/BUL/BUL0870.pdf",

    # Apple
    "Apple_Disease_Compendium.pdf":
        "https://extension.tennessee.edu/publications/Documents/W316.pdf",
    "Apple_IPM_Guide_WSU.pdf":
        "https://treefruit.wsu.edu/wp-content/uploads/2021/02/Apple-IPM.pdf",

    # Grape
    "Grape_Disease_Management_Cornell.pdf":
        "https://ecommons.cornell.edu/bitstream/handle/1813/67156/Grapes_IPM_2020.pdf",
    "Grape_Production_UC_Davis.pdf":
        "https://anrcatalog.ucanr.edu/pdf/21577.pdf",
    "Grape_Powdery_Mildew_UC.pdf":
        "https://ipm.ucanr.edu/pdf/pestnotes/grapepowderymildew.pdf",

    # Pepper
    "Pepper_Bacterial_Spot_UC.pdf":
        "https://ipm.ucanr.edu/pdf/pestnotes/pepperbacterialspot.pdf",
    "Pepper_IPM_Florida.pdf":
        "https://edis.ifas.ufl.edu/pdffiles/HS/HS132400.pdf",

    # Soil, Irrigation, Fertility
    "FAO_Fertilizer_Use_Manual.pdf":
        "https://www.fao.org/3/w2598e/w2598e00.pdf",
    "FAO_Irrigation_Management.pdf":
        "https://www.fao.org/3/s2022e/s2022e.pdf",

    # Pest Management
    "IPM_Handbook_UC.pdf":
        "https://ucanr.edu/sites/fruitreport/files/155619.pdf",
    "FAO_Biological_Control.pdf":
        "https://www.fao.org/3/i8566en/I8566EN.pdf",
    "Crop_Pest_Management_FAO.pdf":
        "https://www.fao.org/3/AC122E/AC122E.pdf",

    # General Crop Guides
    "Cornell_Vegetable_Production_Guide.pdf":
        "https://cpb-us-e1.wpmucdn.com/blogs.cornell.edu/dist/0/726/files/2015/05/2015-Cornell-Vegetable-Guide-1j5q2p9.pdf",
    "Fruit_Production_Manual_MSU.pdf":
        "https://www.canr.msu.edu/uploads/234/78914/fruitproduction2006.pdf",

    # Disease Diagnostics
    "Plant_Disease_Diagnostic_Guide_OSU.pdf":
        "https://plantpath.osu.edu/sites/plantpath/files/imce/files/Disease/DiagnosticGuide.pdf",
    "Plant_Disease_Management_Handbook.pdf":
        "https://pnwhandbooks.org/sites/pnwhandbooks/files/plant/plant-disease-management-guide.pdf",

    # Climate & Agriculture
    "Climate_Smart_Agriculture_WorldBank.pdf":
        "https://www.worldbank.org/content/dam/Worldbank/Document/CSA_Guide_2015.pdf",
    "FAO_Sustainable_Agriculture.pdf":
        "https://www.fao.org/3/i3957e/i3957e.pdf"
}

# ----------------------------------------------------------
# Download Function
# ----------------------------------------------------------
def download_file(url, filename):
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    if os.path.exists(filepath):
        print(f"[SKIP] {filename} already exists.")
        return

    print(f"[DOWNLOAD] {filename}")

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download: {url}")
        return

    total = int(response.headers.get("content-length", 0))
    with open(filepath, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(1024):
            size = f.write(data)
            bar.update(size)

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    print("\n📥 Starting bulk PDF download...\n")

    for filename, url in pdf_links.items():
        try:
            download_file(url, filename)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    print("\n✅ Download complete! All PDFs saved in:", DOWNLOAD_DIR)
