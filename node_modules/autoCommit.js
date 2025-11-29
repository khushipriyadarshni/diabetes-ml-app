const { execSync } = require("child_process");
const fs = require("fs");

for (let i = 1; i <= 100; i++) {
    fs.appendFileSync("commitfile.txt", `Commit ${i}\n`);
    execSync("git add .");
    execSync(`git commit -m "Auto commit ${i}"`);
    console.log("Committed:", i);
}
