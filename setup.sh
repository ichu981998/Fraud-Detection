mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor=\"#1A237E\"\n\
backgroundColor=\"#FFFFFF\"\n\
secondaryBackgroundColor=\"#f8f9fa\"\n\
textColor=\"#000000\"\n\
font=\"sans serif\"\n\
" > ~/.streamlit/config.toml
