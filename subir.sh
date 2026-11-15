#!/usr/bin/env bash
set -e

# Si no se proporciona un mensaje como argumento, usar la fecha y hora actual
if [ -z "$1" ]; then
    MENSAJE="Actualización automática: $(date '+%Y-%m-%d %H:%M:%S')"
else
    MENSAJE="$*"
fi

echo "📦 Agregando archivos al área de preparación (staging)..."
git add .

if git diff --cached --quiet; then
    echo "ℹ️ No hay cambios pendientes para confirmar (commit)."
else
    echo "✅ Creando commit con mensaje: \"$MENSAJE\""
    git commit -m "$MENSAJE"
fi

echo "🚀 Subiendo cambios a origin main..."
git push origin main
