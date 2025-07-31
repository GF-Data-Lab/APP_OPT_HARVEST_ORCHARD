from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# 1) Cargar KML con GDAL/OGR
kml_path = Path("parcela.kml")

gdf = gpd.read_file(kml_path, driver="KML")  # requiere GDAL + libkml
print(f"Filas leídas: {len(gdf)}")
print(gdf.geometry.type.value_counts())

if len(gdf) == 0:
    raise RuntimeError(
        "GeoDataFrame vacío ➜ tu instalación de GDAL/Fiona no tiene soporte KML.\n"
        "Instala libkml o usa el método basado en fastkml."
    )

# 2) (opcional) Dividir MultiGeometry en partes para verlas todas
gdf = gdf.explode(index_parts=False, ignore_index=True)

# 3) Dibujar
ax = gdf.plot(facecolor="none", edgecolor="red", linewidth=1)
ax.set_aspect("equal")
ax.set_title("Parcelas del KML")
plt.tight_layout()
plt.show()           # ← imprescindible si NO estás en Jupyter / IPython
