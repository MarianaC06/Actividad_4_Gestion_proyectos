# Actividad_4_Modelo_preentrenado

## Caso de uso
Clasificación binaria de reseñas IMDb en inglés. Se comparan DistilBERT, BERT y RoBERTa, todos ajustados previamente a SST-2, **sin entrenamiento adicional**. Se mide transferencia de dominio SST-2 → IMDb.

## Ecosistema Hugging Face
- **Models:** pesos, tokenizadores, configuraciones y model cards de los tres modelos.
- **Datasets:** descarga y preparación reproducible de `stanfordnlp/imdb`.
- **Spaces:** plataforma para publicar posteriormente una demo con Gradio; no se despliega en esta entrega.

## Entorno
Google Colab, GPU T4 o similar (Runtime > Change runtime type > T4 GPU). Python 3.10+ recomendado. Requiere Internet para descargar pesos y datos. El login de Hugging Face es opcional para repositorios públicos; el cuaderno permite autenticación interactiva sin exponer tokens.

## Ejecución
1. Abrir `notebooks/benchmark_sentiment.ipynb` en Colab y activar GPU.
2. Ejecutar las celdas en orden; la primera instala dependencias.
3. Si se requiere autenticación, configurar `HF_TOKEN` en los secretos de Colab y ejecutar el login opcional.
4. Descargar el dataset, evaluar los tres modelos y guardar archivos en `results/`.
5. Descargar los resultados de Colab o sincronizarlos con GitHub; el sistema de archivos temporal de Colab no persiste.
6. Alternativamente ejecutar `python src/prepare_data.py` desde la raíz del repositorio para generar la muestra.

## Diseño experimental
300 reseñas del test IMDb, balanceadas 150/150, semilla 42, longitud máxima 256 tokens, batch 16. Misma muestra y configuración para los tres modelos. Warm-up previo; sincronización CUDA; se mide tiempo de inferencia + tokenización, excluyendo carga de modelos y descargas. Accuracy, precision, recall y F1 binarios (clase positiva=1). Solo inferencia: no se modifican los pesos.

## Conclusiones
Se selecciona DistilBERT para primer prototipo, dado que alcanza F1 >= 0.85, recall positivo >= 0.85, y latencia <= 30 ms/reseña en GPU objetivo. 

## Estructura
`notebooks/`, `src/`, `data/`, `results/`, `docs/`, `requirements.txt`.

## Riesgos
Dominio de entrenamiento SST-2 distinto de IMDb; textos truncados a 256 tokens; solo inglés; resultados sensibles al hardware, versión de paquetes y tamaño de muestra. Revisar licencias y condiciones de uso antes de producción.
