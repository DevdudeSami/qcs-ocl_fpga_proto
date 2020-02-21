const fs = require('fs')

if(process.argv[2] === undefined) {
	console.log("No file specified.")
	process.exit()
}

const file = process.argv[2]
console.log(`Compiling file ${file}...`)

const gateCodes = {
	h: 0,
	cnot: 1,
	x: 2,
	y: 3,
	z: 4
}

const gateSizes = {
	h: 1,
	cnot: 2,
	x: 1,
	y: 1,
	z: 1
} 

// read the file
const gir = fs.readFileSync(file, { encoding: 'utf8' }).trim().split("\n")

const n = parseInt(gir.shift())    // number of qubits
const N = 2**n
const G = gir.length

let gateOccurences = {  
	h: 0,
	cnot: 0,
	x: 0,
	y: 0,
	z: 0
}

let multipliers = ["single", "double"]
let multiplierOccurences = {
	single: 0,
	double: 0
}

let includedMultiplierKernels = []
let kernelDefinitions = ""
let kernelNames = ""
let kernelCounts = ""
let includedKernels = ["main", "output"]
let channels = ""

let problemSize = 0 // length of the array input to the FPGA
let problemDefinition = "{"

let multiplierKernelsCode = ""

const singleMultiplierKernelTemplate = fs.readFileSync('templates/singleMultiplierKernel.cl', { encoding: 'utf8' })
function singleMultiplierKernelCode(n) {
	return parseTemplate(singleMultiplierKernelTemplate, { n })
}
const doubleMultiplierKernelTemplate = fs.readFileSync('templates/doubleMultiplierKernel.cl', { encoding: 'utf8' })
function doubleMultiplierKernelCode(n) {
	return parseTemplate(doubleMultiplierKernelTemplate, { n })
}
const singleQubitGateHandlerTemplate = fs.readFileSync('templates/singleQubitGateHandler.cl', { encoding: 'utf8' })
function singleQubitGateHandlerCode(i) {
	return parseTemplate(singleQubitGateHandlerTemplate, { i })
}

for(const line of gir) {
	const gateCall = line.trim().split(" ")
	
 	const controlled = gateCall[0].toLowerCase() === "c"
	const controlQubits = []
	if(controlled) {
		gateCall.shift()

		let controlQubit = parseInt(gateCall[0])

		while(controlQubit == controlQubit) {
			gateCall.shift()
			controlQubits.push(controlQubit)
			controlQubit = parseInt(gateCall[0])
		}
	}

	const gate = gateCall[0]
	const gateSize = gateSizes[gate]

	gateOccurences[gate] += 1
	gateCall[0] = gateCodes[gateCall[0]]

	const multiplier = multipliers[gateSize-1]
	if(!includedMultiplierKernels.includes(multiplier)) {
		const kernelCount = 2**(n-gateSize);
		for(let i = 0; i < kernelCount; i++) {
			if(gateSize == 1) multiplierKernelsCode += singleMultiplierKernelCode(i)
			else if(gateSize == 2) multiplierKernelsCode += doubleMultiplierKernelCode(i)
			kernelDefinitions += `#define K_${multiplier.toUpperCase()}_${i} ${includedKernels.length}\n`
			kernelCounts += `1,\n`
			const kernelName = `${multiplier}Multiplier${i}`
			includedKernels.push(kernelName)
			kernelNames += `"${kernelName}",\n`

			channels += `channel int ${multiplier}Multiplier${i}GateCodeCh;\n`
			channels += `channel cvec2 ${multiplier}Multiplier${i}InCh;\n`
			channels += `channel cvec2 ${multiplier}Multiplier${i}OutCh;\n`	
		} 
		includedMultiplierKernels.push(multiplier)		
	}
	multiplierOccurences[multiplier] += 1

	problemSize += gateCall.length

	problemDefinition += gateCall.join(", ")
	
	problemSize += controlQubits.length + 1
	problemDefinition += `, ${controlQubits.length}` + (controlled ? ", " : "")
	problemDefinition += controlQubits.join(", ")
	
	problemDefinition += ", "
}

problemDefinition = problemDefinition.slice(0, -2) + "}"

let multiplierEachCalls = ""
for(const [multiplier, occurences] of Object.entries(multiplierOccurences)) {
	multiplierEachCalls += `#define K_${multiplier.toUpperCase()}_MULTIPLIER_EACH_CALLS ${occurences}\n`
}

let state = "{(cfloat)(1,0),"
for(let i = 0; i < N-1; i++) state += "(cfloat)(0,0),"
state = state.slice(0,-1) + "}"

const singleQubitGateLoopCount = 2**(n-1)
let singleQubitGateHandlers = ""
for(let i = 0; i < singleQubitGateLoopCount; i++) {
	singleQubitGateHandlers += singleQubitGateHandlerCode(i)
}

const templateInputs = {
	n,
	N,
	G,
	kernelTypeCount: includedKernels.length,
	kernelDefinitions,
	kernelNames,
	kernelCounts,
	multiplierKernelsCode,
	problemSize,
	problemDefinition,
	multiplierEachCalls,
	channels,
	state,
	singleQubitGateLoopCount,
	singleQubitGateHandlers
}

// PROCESS THE TEMPLATE FILES

// read in the template
let mainTemplate = fs.readFileSync('templates/main.cpp', { encoding: 'utf8' })
let kernelsTemplate = fs.readFileSync('templates/kernels.cl', { encoding: 'utf8' })

const parsedMain = parseTemplate(mainTemplate, templateInputs)
const parsedKernels = parseTemplate(kernelsTemplate, templateInputs)

console.log("Writing main.cpp...")
fs.writeFile('./generated/host/main.cpp', parsedMain, { flag: 'w' }, err => {
	if(err) throw err
	console.log("Generated!")
})

console.log("Writing kernels.cl...")
fs.writeFile('./generated/device/kernels.cl', parsedKernels, { flag: 'w' }, err => {
	if(err) throw err
	console.log("Generated!")
})

function parseTemplate(template, data) {
	let result = template
	for (const [k, v] of Object.entries(data)) {
		result = result.replace(new RegExp(`{{${k}}}`, "g"), v)
	}

	return result
}

// function parseTemplates(templates, data) {
// 	let result = [...templates]
// 	for (const [k, v] of Object.entries(data)) {
// 		result.map(t => t.replace(`{{${k}}}`, v))
// 	}

// 	console.log(result)

// 	return result	
// }