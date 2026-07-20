FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.0
LABEL org.opencontainers.image.source="https://github.com/cmk77/gemma-rag-harness" \
      org.opencontainers.image.description="Elasticsearch 8.13 + nori 한국어 형태소 분석기"
RUN bin/elasticsearch-plugin install --batch analysis-nori
