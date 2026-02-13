# Stage 1: Get the Production version of Vue
FROM node:20.18-alpine as build

WORKDIR /bigsmiles-server

RUN npm init -y
RUN npm i express
RUN npm i @cript-web/bigsmiles-toolkit @rdkit/rdkit

COPY ./server.js ./
COPY ./package.json ./

EXPOSE 3318

CMD ["node", "server.js"]
