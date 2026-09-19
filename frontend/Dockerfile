# syntax=docker/dockerfile:1

FROM node:20-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-bookworm-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_REPOMIND_API_URL=http://127.0.0.1:8000/api/v1
ENV NEXT_PUBLIC_REPOMIND_API_URL=$NEXT_PUBLIC_REPOMIND_API_URL
RUN npm run build

FROM node:20-bookworm-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
RUN groupadd --system repomind && useradd --system --gid repomind repomind
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
USER repomind
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
CMD ["node", "server.js"]
