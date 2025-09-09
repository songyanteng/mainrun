try {
  let email
  const envPath = '../.env'
  
  if (await fs.pathExists(envPath)) {
    const envContent = await fs.readFile(envPath, 'utf-8')
    const match = envContent.match(/EMAIL=(.+)/)
    if (match) {
      email = match[1].trim()
      console.log(`Using email: ${email}`)
    }
  }
  
  if (!email) {
    email = await question('Please enter your email address: ')
    
    await fs.writeFile(envPath, `EMAIL=${email}\n`)
    console.log('Email saved for future submissions')
  }
  
  console.log('\n' + '='.repeat(60))
  console.log('LEGAL NOTICE')
  console.log('='.repeat(60))
  console.log('\nBy submitting this assessment, you agree that:')
  console.log('- All submitted code becomes the property of Maincode Pty Ltd')
  console.log('- You assign all intellectual property rights to Maincode Pty Ltd')
  console.log('- You have read and agree to the full legal terms')
  console.log('\nFull terms: https://github.com/maincodehq/mainrun/blob/main/LEGAL-NOTICE.md')
  console.log('='.repeat(60) + '\n')
  
  const confirmation = await question('Do you agree to these terms and want to proceed? (yes/no): ')
  
  if (confirmation.toLowerCase() !== 'yes') {
    console.log('Submission cancelled.')
    process.exit(0)
  }
  
  console.log('\nCreating submission zip...')
  
  await $`cd .. && zip -r submission.zip . -x "node_modules/*" -x "mainrun/data/*"`
  
  console.log('Requesting upload URL...')
  
  const response = await $`curl -s "https://api.hanger.maincode.com/api/v1/upload/request?email=${email}&filename=submission.zip"`
  const uploadUrl = response.stdout.trim()
  
  if (!uploadUrl || uploadUrl.includes('error')) {
    throw new Error(`Failed to get upload URL: ${uploadUrl}`)
  }
  
  console.log('Uploading submission...')
  
  await $`echo ${uploadUrl} | xargs -I {} curl -X PUT {} --upload-file ../submission.zip`
  
  await $`rm -f ../submission.zip`
  
  console.log('✓ Submission uploaded successfully!')
  
} catch (error) {
  await $`rm -f ../submission.zip`.catch(() => {})
  
  console.error('Failed to submit:', error.message)
  process.exit(1)
}