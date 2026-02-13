// server.js

// 1. Import Express and your npm package
import express from 'express';
import * as toolkit from '@cript-web/bigsmiles-toolkit';
import RDKit from '@rdkit/rdkit';

// 2. Initialize the app
const app = express();
const PORT = 3318;
const rdkit = await RDKit();

// Middleware to parse JSON request bodies
app.use(express.json());

// 3. Define the endpoint (The core function lives here)
app.post('/api/molblock-to-bigsmiles', (req, res) => {
    const molblockString = req.body.molblock_string;

    try {
        // Call the function from the npm package
        // console.log(molblockString)
        const bigsmiles = toolkit.molfile_to_bigsmiles(molblockString)
        // console.log(bigsmiles)

        // Send the result back as a JSON response
        res.status(200).json({
            success: true,
            data: bigsmiles
        });

    } catch (error) {
        console.error("Error processing request:", error);
        res.status(500).json({
            success: false,
            message: "Failed to process data using toolkit.molfile_to_bigsmiles()."
        });
    }
});

app.post('/api/bigsmiles-to-molblock', async (req, res) => {
    const bigsmiles = req.body.bigsmiles;

    try {
        // Call the function from the npm package
        const molblock = toolkit.bigsmiles_to_molfile(rdkit, bigsmiles)

        // Send the result back as a JSON response
        res.status(200).json({
            success: true,
            data: molblock
        });

    } catch (error) {
        console.error("Error processing request:", error);
        res.status(500).json({
            success: false,
            message: "Failed to process data using toolkit.bigsmiles_to_molfile()."
        });
    }
});

// 4. Start the server
app.listen(PORT, () => {
    console.log(`Server listening at http://localhost:${PORT}`);
});
