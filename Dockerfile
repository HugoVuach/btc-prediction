# Tant que je developpe pas réélement besoin du Docker
# Mais ca fait une initialisation propre pour la futur expédition du code ou le deploiement sur un serveur cloud AWS, Azure, GCP, etc.

# Étape 1 : utiliser une image Python officielle comme base
FROM python:3.10-slim

# Étape 2 : définir le répertoire de travail
WORKDIR /app

# Étape 3 : copier les fichiers du projet dans le conteneur
COPY . /app

# Étape 4 : installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : exposer un port (utile plus tard si tu ajoutes un dashboard)
EXPOSE 8501

# Étape 6 : démarrer le bot
CMD ["python", "main.py"]
