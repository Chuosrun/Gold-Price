# exports the report to PDF with Microsoft Word (needs Word installed)
$base = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open("$base\Final_Project_Report_Gold_Price_Simple.docx")
$doc.Repaginate()
$pages = $doc.ComputeStatistics(2)
$doc.ExportAsFixedFormat("$base\evidence\preview_simple.pdf", 17)
$doc.Close($false)
$word.Quit()
"pages: $pages" | Out-File -Encoding ascii "$base\evidence\pages.txt"
