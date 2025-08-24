# Galis

## Setting Up the Project Locally

### Prerequisites
- <strong>Docker</strong> 
- <strong>Python 3.9</strong> or higher
- <strong>Poetry</strong> tool for dependency management and packaging
  
### Step 1:

Install required dependencies with ```Makefile```

```bash
make install
```

### Step 2:

Run your <strong>Docker Desktop</strong> application

### Step 3:

This command builds a Docker image for the application.

```bash
make docker-build
```

### Step 4:

This command runs the application inside a Docker container

```bash
make docker-run
```

or you can rung the application locally withour docker container

```bash
make run-app
```