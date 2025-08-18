# 1️⃣ Install poppler-utils (pdftohtml) + qpdf (to remove restrictions)
!apt-get update -qq
!apt-get install -y poppler-utils qpdf

# 2️⃣ Set your file paths
input_pdf = "/content/revolt.pdf"          # 🔹 Change this to your PDF
unlocked_pdf = "/content/unlocked.pdf"
output_dir = "/content/html_output"

# 3️⃣ Remove PDF restrictions
!qpdf --decrypt "$input_pdf" "$unlocked_pdf"

# 4️⃣ Convert unlocked PDF to HTML
import os
os.makedirs(output_dir, exist_ok=True)

# -c = preserve layout, -noframes = single HTML
!pdftohtml -c -noframes "$unlocked_pdf" "$output_dir/output.html"

# 5️⃣ Preview HTML inside Colab
from IPython.display import HTML
HTML(filename=f"{output_dir}/output.html")

# 6️⃣ List output files
!ls -lh "$output_dir"
