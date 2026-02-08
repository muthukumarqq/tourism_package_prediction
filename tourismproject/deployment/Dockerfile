# Base image with Python 3.9
FROM python:3.9-slim

# Create a non-root user
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch to non-root user
USER user

# Set working directory
WORKDIR $HOME/app

# Copy project files with correct ownership
COPY --chown=user . $HOME/app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
