# Use the official Node.js LTS image.
FROM node:18-alpine

# Set the working directory in the container.
WORKDIR /app

# Copy package files and install dependencies.
COPY package*.json ./
RUN npm install

# Copy the remaining application code.
COPY . .

# Expose the port used by Next.js (default is 3000).
EXPOSE 3000

# Start the Next.js development server.
CMD ["npm", "run", "dev"]
