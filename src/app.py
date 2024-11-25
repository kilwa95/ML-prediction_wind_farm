from components.header import render_header
from components.sidebar import render_sidebar
from components.home import render_home
from components.data_visualization import render_data_visualization

def main():
    render_header("Mon Application Streamlit")
    selected = render_sidebar()

    if selected == "Accueil":
        render_home()

    if selected == "Visualisation des Données":
        render_data_visualization()
  

if __name__ == "__main__":
    main() 
