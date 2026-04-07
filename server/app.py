import uvicorn

# This is the boilerplate entry point the OpenEnv linter demands.
# It simply points back to your actual root server.py file.
def main():
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()