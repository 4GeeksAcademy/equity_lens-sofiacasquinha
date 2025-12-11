DROP VIEW IF EXISTS vw_model_cohort;

CREATE VIEW vw_model_cohort AS
SELECT
    year,
    lnrwg,
    uhrswork,
    annhrs,
    sex,
    age,
    educ99,
    ba,
    adv,
    potexp,
    potexp2,
    classwkr,
    "union",
    ft,
    
     -- occupation dummies
    manager, business, financialop, computer,
    architect, scientist, socialworker,
    postseceduc, legaleduc, artist,
    lawyerphysician, healthcare, healthsupport,
    protective, foodcare, building,
    sales, officeadmin, farmer,
    constructextractinstall, production, transport_occ,
    
    -- industry dummies
    finance, medical, education, publicadmin,
    professional, durables, nondurables,
    retailtrade, wholesaletrade,
    transport_ind, utilities, communications,
    socartother, hotelsrestaurants,
    agriculture, miningconstruction

FROM cps_raw
WHERE
    -- valid wages and hours
    lnrwg   IS NOT NULL AND lnrwg > 0 -- log real hourly wage (target)
    AND uhrswork    IS NOT NULL AND uhrswork  > 10  AND uhrswork <= 100
    AND annhrs  >= uhrswork * 50

    -- demographics
    AND age BETWEEN 25 AND 64
    AND sex  IN (1, 2)

    -- education and experience
    AND educ99  IS NOT NULL
    AND potexp  IS NOT NULL
    AND potexp2 IS NOT NULL

    -- job status fields
    AND classwkr    IS NOT NULL
    AND "union" IS NOT NULL
    AND ft  IS NOT NULL

    -- exactly one occupation and one industry per worker
    AND (manager + business + financialop + computer +
        architect + scientist + socialworker +
        postseceduc + legaleduc + artist +
        lawyerphysician + healthcare + healthsupport +
        protective + foodcare + building +
        sales + officeadmin + farmer +
        constructextractinstall + production + transport_occ) = 1

    AND (finance + medical + education + publicadmin +
        professional + durables + nondurables +
        retailtrade + wholesaletrade +
        transport_ind + utilities + communications +
        socartother + hotelsrestaurants +
        agriculture + miningconstruction) = 1
  
