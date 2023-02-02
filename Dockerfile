FROM google-cloud-cli:latest
RUN pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN echo "Connected OK"