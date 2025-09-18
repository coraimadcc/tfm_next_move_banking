## Datos utilizados

Los datos empleados en este proyecto corresponden al **Multimodal Bank Dataset (MBD)** publicado por **Sberbank AI Lab** y disponible públicamente en Hugging Face:

- [MBD (versión completa)](https://huggingface.co/datasets/ai-lab/MBD)  
- [MBD-mini (versión reducida)](https://huggingface.co/datasets/ai-lab/MBD-mini)

⚠️ **Nota importante**:  
Este repositorio **no incluye los archivos de datos** debido a restricciones de tamaño y buenas prácticas de versionado.  
Para reproducir los experimentos, se recomienda descargar los datos directamente desde Hugging Face y almacenarlos localmente o en un servicio de nube (por ejemplo, Azure Blob Storage o Google Drive). 

### Configuración de credenciales

Este proyecto requiere que las credenciales se gestionen mediante **Databricks Secrets**.  
Se espera que el usuario configure un *secret scope* llamado `tfm_next_move_banking` con las siguientes claves:

- `openai_api`: API Key de OpenAI  
- `telegram`: Token del bot de Telegram  
- `chat_id`: ID del chat de destino  

⚠️ **Nota**: Los valores de estas credenciales no están incluidos en este repositorio por motivos de seguridad.

