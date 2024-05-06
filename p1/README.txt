Para ejecutar es necesario tener dry_bean_classifier.jl y el DataSet en el mismo directorio. Si hay problemas de dependencias, para instalar las librerias descomentar el principio del código.

plots-> Carpeta en la que se almacenan las imagenes generadas durante la ejecución

dry_bean_classifier.jl -> Archivo principal de la ejecución

fucntions.jl -> Funciones auxiliares para la ejecución del archivo principal

Dry_Bean_Dataset.xlsx -> Base de Datos

index.txt -> Archivo de índices con los índices utilizados en la validación cruzada, para repetir los
experimentos. El codigo ademas contiene una seed para repetir los experimentos.

results.csv -> Archivo que almacena los resultados de las ejecuciones de los modelos como: (Precisión Media, Std Precisión)
